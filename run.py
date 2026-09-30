"""Run any generated QSTS case selected explicitly with --data-dir."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.power_flow import NetworkModel
from src.profile_store import ProfileStore
from src.qsts import run_qsts
from src.solvers.opendss_solver import OpenDSSSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a generated OpenDSS QSTS case")
    parser.add_argument("--data-dir", type=Path, required=True, help="Case directory containing network.json and network.dss.")
    parser.add_argument("--output", type=Path, help="Result directory; defaults to output/<case-directory-name>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    required_files = [data_dir / "network.json", data_dir / "network.dss", data_dir / "load_profiles.csv"]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Case input data is missing:\n" + "\n".join(missing))

    output_dir = args.output.resolve() if args.output else ROOT / "output" / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    network = NetworkModel.from_json(data_dir / "network.json")
    solver = OpenDSSSolver(data_dir / "network.dss")
    solver.build(network)
    profiles = ProfileStore.from_case(network, data_dir)

    qsts_bus, qsts_system, qsts_branches = run_qsts(solver, profiles, network)
    qsts_bus.to_csv(output_dir / "qsts_bus_voltages.csv", index=False)
    qsts_system.to_csv(output_dir / "qsts_system.csv", index=False)
    qsts_branches.to_csv(output_dir / "qsts_branch_results.csv", index=False)
    constraint_columns = [
        "timestamp", "voltage_low_count", "voltage_high_count", "voltage_unbalance_count",
        "line_thermal_violation_count", "transformer_thermal_violation_count",
        "max_voltage_unbalance_pct", "max_line_loading_pct", "max_transformer_loading_pct",
        "reverse_power_flow", "reverse_power_kw",
    ]
    hosting_columns = [
        "timestamp", "voltage_headroom_pu", "line_loading_headroom_pct",
        "transformer_loading_headroom_pct", "binding_constraint",
    ]
    qsts_system.loc[:, constraint_columns].to_csv(output_dir / "qsts_constraints.csv", index=False)
    qsts_system.loc[:, hosting_columns].to_csv(output_dir / "qsts_hosting_capacity_metrics.csv", index=False)
    print(f"QSTS completed for case: {data_dir.name}")
    print(f"Input data: {data_dir}")
    print(f"Results: {output_dir.resolve()}")
    print(qsts_system[["timestamp", "total_load_kw", "min_v_pu", "max_v_pu", "loss_kw"]].to_string(index=False))


if __name__ == "__main__":
    main()
