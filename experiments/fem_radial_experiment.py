"""Generate and compare a long, unbalanced three-phase FE feeder with OpenDSS.

Run from repository root: python experiments/fem_radial_experiment.py
This intentionally avoids the IEEE-13 case's transformers, regulator,
switches, shunt capacitors, delta devices and non-ideal source.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
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


def generate_case(folder: Path, n_segments: int, hours: int) -> None:
    if n_segments < 3 or hours < 1:
        raise ValueError("Need at least three segments and one hour.")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "der_profiles").mkdir(exist_ok=True)
    start = datetime(2026, 1, 1)
    buses = [{"id": "source", "phases": [1, 2, 3], "nominal_kv_ll": 12.47, "is_slack": True}]
    buses += [{"id": f"b{i}", "phases": [1, 2, 3], "nominal_kv_ll": 12.47} for i in range(1, n_segments + 1)]
    r = [[0.34, 0.035, 0.03], [0.035, 0.36, 0.032], [0.03, 0.032, 0.33]]
    x = [[0.28, 0.065, 0.06], [0.065, 0.29, 0.063], [0.06, 0.063, 0.27]]
    line_code = {"unit": "km", "phases": [1, 2, 3], "r_ohm_per_unit": r, "x_ohm_per_unit": x}
    branches = [
        {"id": f"L{i}", "kind": "line", "from_bus": "source" if i == 1 else f"b{i-1}",
         "to_bus": f"b{i}", "phases": [1, 2, 3], "length": 0.12,
         "line_code": "main", "normal_amps": 400.0}
        for i in range(1, n_segments + 1)
    ]
    devices: list[dict] = []
    load_rows: list[dict] = []
    dss_lines = [
        "Clear", "Set DefaultBaseFrequency=60",
        "New Circuit.FEMFeeder basekv=12.47 pu=1 phases=3 bus1=source angle=0",
        "Edit Vsource.Source R1=0.0000001 X1=0.0000001 R0=0.0000001 X0=0.0000001",
        "New LineCode.main nphases=3 units=km",
        "~ rmatrix=[0.34 | 0.035 0.36 | 0.03 0.032 0.33]",
        "~ xmatrix=[0.28 | 0.065 0.29 | 0.06 0.063 0.27]",
        "~ cmatrix=[0 | 0 0 | 0 0 0]",
    ]
    for branch in branches:
        dss_lines.append(f"New Line.{branch['id']} phases=3 bus1={branch['from_bus']}.1.2.3 bus2={branch['to_bus']}.1.2.3 linecode=main length=0.12 units=km")
    for i in range(1, n_segments + 1):
        for phase in (1, 2, 3):
            name = f"Load_{i}_{phase}"
            devices.append({"id": name, "kind": "load", "bus": f"b{i}", "phases": [phase], "connection": "wye", "kv": 12.47 / np.sqrt(3)})
            dss_lines.append(f"New Load.{name} phases=1 bus1=b{i}.{phase} conn=wye model=1 kV={12.47/np.sqrt(3):.12g} kW=0 kvar=0")
            for hour in range(hours):
                time = start + timedelta(hours=hour)
                shape = 0.75 + 0.15 * np.sin(2.0 * np.pi * (hour - 6) / 24.0)
                unbalance = {1: 1.15, 2: 0.9, 3: 0.8}[phase]
                p = (2.3 + 0.2 * (i % 5)) * shape * unbalance
                load_rows.append({"timestamp": time.isoformat(sep=" "), "load_name": name, "p_kw": round(p, 7), "q_kvar": round(p * 0.3, 7)})
    for i in range(3, n_segments + 1, 3):
        phase = i % 3 + 1
        name = f"PV_{i}"
        path = f"der_profiles/{name.lower()}.csv"
        devices.append({"id": name, "kind": "pv", "bus": f"b{i}", "phases": [phase], "connection": "wye", "kv": 12.47 / np.sqrt(3), "p_rated_kw": 5.0, "profile_file": path})
        dss_lines.append(f"New Generator.{name} phases=1 bus1=b{i}.{phase} conn=wye model=1 kV={12.47/np.sqrt(3):.12g} kW=0 kvar=0")
        pd.DataFrame([{
            "timestamp": (start + timedelta(hours=hour)).isoformat(sep=" "),
            "pv_kw": round(max(0.0, 4.2 * np.sin(np.pi * (hour - 6) / 12.0)), 7),
            "pv_kvar": 0.0, "controller_type": "volt_var",
            "q_limit_kvar": 1.5, "v_ref_pu": 1.0, "droop_kvar_per_pu": 30.0,
        } for hour in range(hours)]).to_csv(folder / path, index=False)
    for i in range(8, n_segments + 1, 8):
        phase = (i + 1) % 3 + 1
        name = f"EV_{i}"
        path = f"der_profiles/{name.lower()}.csv"
        devices.append({"id": name, "kind": "ev", "bus": f"b{i}", "phases": [phase], "connection": "wye", "kv": 12.47 / np.sqrt(3), "p_rated_kw": 3.0, "profile_file": path})
        dss_lines.append(f"New Load.{name} phases=1 bus1=b{i}.{phase} conn=wye model=1 kV={12.47/np.sqrt(3):.12g} kW=0 kvar=0")
        pd.DataFrame([{
            "timestamp": (start + timedelta(hours=hour)).isoformat(sep=" "),
            "ev_kw": 2.0 if hour >= 17 or hour < 6 else 0.6,
            "ev_kvar": 0.0, "controller_type": "constant_pq",
        } for hour in range(hours)]).to_csv(folder / path, index=False)
    model = {
        "schema_version": "1.0", "base": {"frequency_hz": 60, "base_kv_ll": 12.47, "slack_bus": "source"},
        "source": {"bus": "source", "phases": [1, 2, 3], "base_kv_ll": 12.47, "voltage_pu": 1.0, "angle_deg": 0.0},
        "simulation": {"start_datetime": start.isoformat(sep=" "), "end_datetime": (start + timedelta(hours=hours)).isoformat(sep=" "), "step_seconds": 3600},
        "buses": buses, "line_codes": {"main": line_code}, "branches": branches,
        "shunt_devices": [], "devices": devices,
        "constraints": {"voltage_min_pu": 0.95, "voltage_max_pu": 1.05},
    }
    (folder / "network.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(load_rows).to_csv(folder / "load_profiles.csv", index=False)
    dss_lines += ["Set VoltageBases=[12.47]", "CalcVoltageBases", "Set MaxIterations=100", "Set ControlMode=off", "Solve"]
    (folder / "network.dss").write_text("\n".join(dss_lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    case = ROOT / "work" / "fem_radial_case"
    output = ROOT / "output" / "fem_radial_experiment"
    output.mkdir(parents=True, exist_ok=True)
    generate_case(case, args.segments, args.hours)
    network = NetworkModel.from_json(case / "network.json")
    timings: dict[str, dict[str, float]] = {}
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for label, solver in (
        ("fem", FiniteElementPFSolver()),
        ("opendss", OpenDSSSolver(case / "network.dss")),
    ):
        start = perf_counter()
        solver.build(network)
        build_seconds = perf_counter() - start
        profiles = ProfileStore.from_case(network, case)
        start = perf_counter()
        results[label] = run_qsts(solver, profiles, network)
        solve_seconds = perf_counter() - start
        timings[label] = {"build_seconds": build_seconds, "qsts_seconds": solve_seconds}
        for name, table in zip(("bus", "system", "branches"), results[label], strict=True):
            table.to_csv(output / f"{label}_{name}.csv", index=False)
    fem_bus, fem_system, _ = results["fem"]
    dss_bus, dss_system, _ = results["opendss"]
    keys = ["timestamp", "bus", "node"]
    joined = fem_bus.merge(dss_bus, on=keys, suffixes=("_fem", "_dss"), validate="one_to_one")
    if len(joined) != len(fem_bus) or len(joined) != len(dss_bus):
        raise RuntimeError("Voltage record sets differ between FEM and OpenDSS.")
    joined["abs_voltage_error_pu"] = (joined["v_pu_fem"] - joined["v_pu_dss"]).abs()
    joined.to_csv(output / "voltage_comparison.csv", index=False)
    summary = {
        "case": "line-only three-phase radial feeder with wye load/PV/EV, nearly ideal OpenDSS source",
        "segments": args.segments, "hours": args.hours, "phase_nodes": len(network.buses) * 3,
        "der_count": len(network.der_indices), "load_count": len(network.load_indices),
        "max_voltage_magnitude_error_pu": float(joined["abs_voltage_error_pu"].max()),
        "mean_voltage_magnitude_error_pu": float(joined["abs_voltage_error_pu"].mean()),
        "max_loss_difference_kw": float((fem_system.loss_kw - dss_system.loss_kw).abs().max()),
        "max_source_p_difference_kw": float((fem_system.source_p_kw - dss_system.source_p_kw).abs().max()),
        "timings": timings,
        "notes": "Timing is one local Python run, excludes CSV writes and cannot establish asymptotic speed superiority. P1 FEM on piecewise-constant unshunted lines equals series Ybus assembly.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=60)
    parser.add_argument("--hours", type=int, default=24)
    run(parser.parse_args())
