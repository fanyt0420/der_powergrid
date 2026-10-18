"""Reproducible comparisons on both complete DER cases.

Run: python -m test.compare_solvers --output output/fem_comparison
The automatic-control comparison uses each production backend unchanged.
The matched-control comparison replays OpenDSS taps and final DER commands in
FEM to separate network/model differences from controller hysteresis.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd

from src.power_flow import NetworkModel, OperatingPoint, PFState
from src.profile_store import ProfileStore
from src.qsts import run_qsts
from src.solvers.fem_solver import FiniteElementPFSolver
from src.solvers.opendss_solver import OpenDSSSolver

ROOT = Path(__file__).resolve().parents[1]
CASES = ("ieee13_unbalanced_der", "ieee13_measured")


class RecordingSolver:
    def __init__(self, backend):
        self.backend = backend
        self.results = []
        self.points = []

    def solve(self, operating_point, state, context):
        result = self.backend.solve(operating_point, state, context)
        self.results.append(result)
        self.points.append(operating_point)
        return result


def _merged_frames(fem, reference, keys):
    left, right = fem.copy(), reference.copy()
    for key in keys:
        if key in {"bus", "branch"}:
            left[key] = left[key].str.lower()
            right[key] = right[key].str.lower()
    joined = left.merge(right, on=keys, suffixes=("_fem", "_opendss"), how="outer", indicator=True, validate="one_to_one")
    if not joined["_merge"].eq("both").all():
        raise AssertionError(f"Unmatched output rows for keys {keys}")
    return joined.drop(columns="_merge")


def compare_case(case: str, output: Path | None = None) -> dict:
    folder = ROOT/"data"/case
    network = NetworkModel.from_json(folder/"network.json")
    frames, recordings, timing = {}, {}, {}
    for label, solver in (("opendss", OpenDSSSolver(folder/"network.dss")), ("fem", FiniteElementPFSolver())):
        profiles = ProfileStore.from_case(network, folder)  # Independent SOC histories.
        start = perf_counter()
        solver.build(network)
        build_seconds = perf_counter()-start
        recorder = RecordingSolver(solver)
        start = perf_counter()
        frames[label] = run_qsts(recorder, profiles, network)
        timing[label] = {"build_seconds": build_seconds, "qsts_seconds": perf_counter()-start}
        recordings[label] = recorder
    buses = _merged_frames(frames["fem"][0], frames["opendss"][0], ["timestamp", "bus", "node"])
    buses["magnitude_error_pu"] = abs(buses.v_pu_fem-buses.v_pu_opendss)
    buses["complex_error_pu"] = abs(buses.v_pu_fem*np.exp(1j*np.deg2rad(buses.angle_deg_fem)) - buses.v_pu_opendss*np.exp(1j*np.deg2rad(buses.angle_deg_opendss)))
    branches = _merged_frames(frames["fem"][2], frames["opendss"][2], ["timestamp", "branch"])
    branches["max_current_difference_a"] = abs(branches.max_current_a_fem-branches.max_current_a_opendss)
    branches["tap_difference_steps"] = abs(branches.tap_position_fem-branches.tap_position_opendss)
    systems = _merged_frames(frames["fem"][1], frames["opendss"][1], ["timestamp"])

    # This diagnostic intentionally receives REFERENCE controls, never reference
    # voltages or matrices. The production FEM path above receives neither.
    fixed = FiniteElementPFSolver(regulator_mode="fixed")
    fixed.build(network)
    matched, state = [], None
    for point, ref in zip(recordings["opendss"].points, recordings["opendss"].results):
        ref_taps = {str(row["branch"]).lower(): int(round(row["tap_position"]))
                    for row in ref.branch_records if row["branch_kind"] == "regulator"}
        state = PFState(voltage_guess=state.voltage_guess if state else {},
                        backend_state={"tap_positions": {name: ref_taps[name.lower()] for name in fixed.regulators}})
        commands = {name: replace(cmd, controller_type="constant_pq") for name, cmd in ref.final_der_commands.items()}
        result = fixed.solve(replace(point, der_commands=commands), state)
        state = result.next_state
        ref_branches = {str(row["branch"]).lower(): row for row in ref.branch_records}
        matched.append({"timestamp": point.time.isoformat(),
                        "max_complex_error_pu": max(abs(v-ref.bus_voltages[k]) for k, v in result.bus_voltages.items()),
                        "max_magnitude_error_pu": max(abs(abs(v)-abs(ref.bus_voltages[k])) for k, v in result.bus_voltages.items()),
                        "max_branch_current_difference_a": max(abs(row["max_current_a"]-ref_branches[str(row["branch"]).lower()]["max_current_a"]) for row in result.branch_records),
                        "loss_difference_kw": abs(result.summary["loss_kw"]-ref.summary["loss_kw"]),
                        "max_kcl_residual_a": result.summary["max_kcl_residual_a"]})
    matched = pd.DataFrame(matched)
    # Hash every electrical input, including redirected DSS files and profiles.
    input_files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".json", ".csv", ".dss"})
    manifest = {str(p.relative_to(folder)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
    soc_columns = [c for c in frames["fem"][1] if c.endswith("_soc_pct")]
    soc_error = max((float(abs(systems[c+"_fem"]-systems[c+"_opendss"]).max()) for c in soc_columns), default=0.)
    summary = {"case": case, "time_steps": len(frames["fem"][1]), "phase_nodes": len(network.buses) and len(recordings["fem"].results[0].bus_voltages),
               "branches": len(network.branches), "der_devices": len(network.der_indices),
               "auto_control": {"max_magnitude_error_pu": float(buses.magnitude_error_pu.max()),
                                "max_complex_error_pu": float(buses.complex_error_pu.max()),
                                "max_branch_current_difference_a": float(branches.max_current_difference_a.max()),
                                "max_tap_difference_steps": float(branches.tap_difference_steps.max()),
                                "max_loss_difference_kw": float(abs(systems.loss_kw_fem-systems.loss_kw_opendss).max()),
                                "max_soc_difference_pct": soc_error},
               "matched_controls": {c: float(matched[c].max()) for c in matched if c != "timestamp"},
               "fem_max_kcl_residual_a": float(frames["fem"][1].max_kcl_residual_a.max()),
               "timing_single_run": timing,
               "timing_note": "Local single run, no speed claim; build and QSTS timed separately; I/O excluded.",
               "inputs_sha256": manifest}
    if output is not None:
        dest = Path(output)/case
        dest.mkdir(parents=True, exist_ok=True)
        buses.to_csv(dest/"voltage_comparison.csv", index=False)
        branches.to_csv(dest/"branch_comparison.csv", index=False)
        systems.to_csv(dest/"system_comparison.csv", index=False)
        matched.to_csv(dest/"matched_control_comparison.csv", index=False)
        (dest/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, action="append", help="May be repeated; defaults to both cases.")
    parser.add_argument("--output", type=Path, default=ROOT/"output"/"fem_comparison")
    args = parser.parse_args()
    import scipy
    import dss
    report = {"environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "dss_python": dss.__version__},
              "cases": [compare_case(case, args.output) for case in args.case or CASES]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in report["cases"]:
        print(json.dumps({k: v for k, v in row.items() if k != "inputs_sha256"}, indent=2))
    print(f"Report: {(args.output/'report.json').resolve()}")


if __name__ == "__main__":
    main()
