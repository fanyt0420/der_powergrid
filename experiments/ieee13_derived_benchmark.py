"""Auditable IEEE-13-derived *line-only* benchmark for the P1 graph FEM solver.

This is a modified subset, not the published full IEEE 13-node feeder. It
preserves selected published line matrices, phase connections, and lengths;
replaces the upstream transformer/regulator by an ideal 4.16-kV source;
omits switch, shunts, delta and transformer-fed loads; and uses project DER
profiles. The original model 5 load at 611 is projected to constant PQ.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.power_flow import NetworkModel
from src.profile_store import ProfileStore
from src.qsts import run_qsts
from src.solvers.fem_solver import FiniteElementPFSolver
from src.solvers.opendss_solver import OpenDSSSolver

SOURCE = ROOT / "data" / "ieee13_unbalanced_der"
BRANCH_IDS = ("650632", "632670", "670671", "671680", "632633", "632645", "645646", "671684", "684611")
LOAD_IDS = ("645", "611", "670a", "670b", "670c")
DER_IDS = ("PV_632_A", "PV_670_C", "PV_671_A", "Wind_680_ABC", "EV_671_C", "EV_611_C")


def environment() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "dss-python"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "os": platform.platform(), "processor": platform.processor(), "packages": packages}


def _triangle(matrix: list[list[float]]) -> str:
    return "[" + " | ".join(" ".join(f"{matrix[i][j]:.12g}" for j in range(i + 1)) for i in range(len(matrix))) + "]"


def generate_case(folder: Path) -> dict:
    raw = json.loads((SOURCE / "network.json").read_text(encoding="utf-8"))
    branch_by_id = {item["id"]: item for item in raw["branches"]}
    device_by_id = {item["id"]: item for item in raw["devices"]}
    branches = [branch_by_id[name] for name in BRANCH_IDS]
    bus_names = {"RG60"} | {branch["from_bus"] for branch in branches} | {branch["to_bus"] for branch in branches}
    buses = [dict(bus, is_slack=(bus["id"] == "RG60")) for bus in raw["buses"] if bus["id"] in bus_names]
    devices = [device_by_id[name] for name in (*LOAD_IDS, *DER_IDS)]
    used_codes = {branch["line_code"] for branch in branches}
    codes = {name: raw["line_codes"][name] for name in sorted(used_codes)}
    case = {
        "schema_version": "1.0",
        "base": {"frequency_hz": 60, "base_kv_ll": 4.16, "slack_bus": "RG60"},
        "source": {"bus": "RG60", "phases": [1, 2, 3], "base_kv_ll": 4.16, "voltage_pu": 1.0, "angle_deg": 0.0},
        "simulation": raw["simulation"], "constraints": raw["constraints"],
        "buses": buses, "branches": branches, "line_codes": codes,
        "shunt_devices": [], "devices": devices,
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "der_profiles").mkdir(exist_ok=True)
    (folder / "network.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    all_loads = pd.read_csv(SOURCE / "load_profiles.csv")
    all_loads[all_loads.load_name.isin(LOAD_IDS)].to_csv(folder / "load_profiles.csv", index=False)
    for name in DER_IDS:
        profile = Path(device_by_id[name]["profile_file"])
        shutil.copyfile(SOURCE / profile, folder / profile)

    lines = [
        "Clear", "Set DefaultBaseFrequency=60",
        "New Circuit.IEEE13Derived basekv=4.16 pu=1 phases=3 bus1=RG60 angle=0",
        "Edit Vsource.Source R1=0.0000001 X1=0.0000001 R0=0.0000001 X0=0.0000001",
    ]
    for name, code in codes.items():
        n = len(code["r_ohm_per_unit"])
        lines.extend((
            f"New LineCode.{name} nphases={n} units={code['unit']}",
            f"~ rmatrix={_triangle(code['r_ohm_per_unit'])}",
            f"~ xmatrix={_triangle(code['x_ohm_per_unit'])}",
            f"~ cmatrix={_triangle([[0.0] * n for _ in range(n)])}",
        ))
    for branch in branches:
        phases = ".".join(str(p) for p in branch["phases"])
        lines.append(
            f"New Line.{branch['id']} phases={len(branch['phases'])} "
            f"bus1={branch['from_bus']}.{phases} bus2={branch['to_bus']}.{phases} "
            f"linecode={branch['line_code']} length={branch['length']} units={branch['length_unit']}"
        )
    for device in devices:
        kind = device["kind"]
        name = device["id"]
        phases = ".".join(str(p) for p in device["phases"])
        dss_kind = "Generator" if kind in {"pv", "wind"} else "Load"
        lines.append(
            f"New {dss_kind}.{name} phases={len(device['phases'])} bus1={device['bus']}.{phases} "
            f"conn=wye model=1 kV={device['kv']:.12g} kW=0 kvar=0 Vminpu=0 Vmaxpu=2"
        )
    lines += ["Set VoltageBases=[4.16]", "CalcVoltageBases", "Set MaxIterations=100", "Set ControlMode=off", "Solve"]
    (folder / "network.dss").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "origin": "IEEE PES 13-node test feeder line data as stored in this repository",
        "origin_url": "https://sites.ieee.org/pes-testfeeders/resources/",
        "classification": "IEEE-13-derived modified line-only subset; not the published full feeder",
        "retained_branches": list(BRANCH_IDS), "retained_loads": list(LOAD_IDS), "retained_der": list(DER_IDS),
        "modifications": [
            "RG60 is an ideal 4.16-kV three-phase source; upstream 115-kV source, transformer, and regulators are omitted",
            "selected IEEE line R/X matrices, phase mappings, and physical lengths are preserved; line charging is set to zero",
            "switch, capacitor, transformer-fed node 634, delta loads and unsupported laterals are omitted",
            "load 611 changes from original OpenDSS model 5 to constant-PQ model 1 at the existing profile power",
            "DER placements and 24-hour profiles are project-authored, not IEEE published measurements",
            "OpenDSS source has tiny nonzero impedance while FEM source is ideal",
        ],
    }


def compare(fem: tuple[pd.DataFrame, ...], dss: tuple[pd.DataFrame, ...]) -> dict:
    a_bus, a_sys, a_branch = fem
    b_bus, b_sys, b_branch = dss
    a_bus, b_bus = a_bus.copy(), b_bus.copy()
    a_bus["bus"] = a_bus.bus.str.lower()
    b_bus["bus"] = b_bus.bus.str.lower()
    voltages = a_bus.merge(b_bus, on=["timestamp", "bus", "node"], suffixes=("_fem", "_dss"), validate="one_to_one")
    if len(voltages) != len(a_bus) or len(voltages) != len(b_bus):
        fem_keys = set(map(tuple, a_bus[["bus", "node"]].drop_duplicates().to_numpy()))
        dss_keys = set(map(tuple, b_bus[["bus", "node"]].drop_duplicates().to_numpy()))
        raise AssertionError(f"Voltage node/time sets differ; only FEM={fem_keys - dss_keys}, only OpenDSS={dss_keys - fem_keys}")
    v_error = (voltages.v_pu_fem - voltages.v_pu_dss).abs()
    angle_diff = (voltages.angle_deg_fem - voltages.angle_deg_dss + 180.0) % 360.0 - 180.0
    a_branch, b_branch = a_branch.copy(), b_branch.copy()
    a_branch["branch"] = a_branch.branch.str.lower()
    b_branch["branch"] = b_branch.branch.str.lower()
    branches = a_branch.merge(b_branch, on=["timestamp", "branch"], suffixes=("_fem", "_dss"), validate="one_to_one")
    if len(branches) != len(a_branch) or len(branches) != len(b_branch):
        raise AssertionError("Branch/time sets differ")
    return {
        "voltage_points": len(voltages),
        "voltage_max_abs_error_pu": float(v_error.max()),
        "voltage_mean_abs_error_pu": float(v_error.mean()),
        "voltage_p95_abs_error_pu": float(v_error.quantile(0.95)),
        "angle_max_abs_error_deg": float(angle_diff.abs().max()),
        "branch_current_max_abs_error_a": float((branches.max_current_a_fem - branches.max_current_a_dss).abs().max()),
        "loss_max_abs_error_kw": float((a_sys.loss_kw - b_sys.loss_kw).abs().max()),
        "source_p_max_abs_error_kw": float((a_sys.source_p_kw - b_sys.source_p_kw).abs().max()),
        "fem_max_pf_iterations_total": int(a_sys.pf_iterations_total.max()),
        "fem_max_pf_solve_count": int(a_sys.pf_solve_count.max()),
        "all_snapshots_converged": bool(a_sys.converged.all() and b_sys.converged.all()),
    }


def run(repeats: int) -> dict:
    if repeats < 3:
        raise ValueError("Use at least three timed repetitions")
    case_dir = ROOT / "work" / "ieee13_derived_fem"
    output_dir = ROOT / "output" / "ieee13_derived_fem"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate_case(case_dir)
    network = NetworkModel.from_json(case_dir / "network.json")
    profiles = ProfileStore.from_case(network, case_dir)
    timings: dict[str, list[dict[str, float]]] = defaultdict(list)
    first: dict[str, tuple[pd.DataFrame, ...]] = {}
    for round_index in range(repeats + 1):
        labels = ("fem", "opendss") if round_index % 2 == 0 else ("opendss", "fem")
        for label in labels:
            solver = FiniteElementPFSolver() if label == "fem" else OpenDSSSolver(case_dir / "network.dss")
            start = perf_counter()
            solver.build(network)
            built = perf_counter()
            result = run_qsts(solver, profiles, network)
            finished = perf_counter()
            if round_index == 0:
                first[label] = result
            else:
                timings[label].append({"build_seconds": built - start, "qsts_seconds": finished - built})
    metrics = compare(first["fem"], first["opendss"])
    summary = {
        "environment": environment(),
        "case": manifest, "bus_count": len(network.buses), "phase_node_count": sum(len(bus["phases"]) for bus in network.buses),
        "branch_count": len(network.branches), "load_count": len(network.load_indices), "der_count": len(network.der_indices),
        "snapshot_count": len(first["fem"][1]), "metrics": metrics, "repetitions": repeats,
        "timing_method": "one warmup then alternating solver order; Python perf_counter; excludes CSV writes; same process and machine",
        "timings": {
            label: {
                phase: {
                    "median_seconds": float(np.median([sample[phase] for sample in timings[label]])),
                    "p25_seconds": float(np.percentile([sample[phase] for sample in timings[label]], 25)),
                    "p75_seconds": float(np.percentile([sample[phase] for sample in timings[label]], 75)),
                    "samples_seconds": [sample[phase] for sample in timings[label]],
                }
                for phase in ("build_seconds", "qsts_seconds")
            }
            for label in ("fem", "opendss")
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, tables in first.items():
        for table_name, table in zip(("bus", "system", "branches"), tables, strict=True):
            table.to_csv(output_dir / f"{label}_{table_name}.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=15)
    run(parser.parse_args().repeats)
