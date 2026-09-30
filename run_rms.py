"""Run a configured seconds-scale RMS scenario from a saved QSTS time point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.power_flow import NetworkModel
from src.rms_model import load_qsts_operating_point, load_rms_config, run_rms
from src.solvers.opendss_solver import OpenDSSSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start an RMS scenario from a saved QSTS operating point")
    parser.add_argument("--data-dir", type=Path, required=True, help="Case directory containing network.json and rms_config.json.")
    parser.add_argument("--qsts-output", type=Path, required=True, help="Directory containing qsts_bus_voltages.csv and qsts_system.csv.")
    parser.add_argument("--time", type=str, required=True, help="QSTS timestamp (YYYY-MM-DD HH:MM:SS) used as the RMS initialization point.")
    parser.add_argument("--scenario", default="flat_run", help="Scenario name in rms_config.json (default: flat_run).")
    parser.add_argument("--output", type=Path, help="Defaults to <qsts-output>/rms/<scenario>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir, qsts_output = args.data_dir.resolve(), args.qsts_output.resolve()
    required = [data_dir / "network.json", data_dir / "network.dss", data_dir / "rms_config.json", qsts_output / "qsts_bus_voltages.csv", qsts_output / "qsts_system.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("RMS input is missing:\n" + "\n".join(missing))
    network = NetworkModel.from_json(data_dir / "network.json")
    rms_config = load_rms_config(data_dir / "rms_config.json")
    operating_point = load_qsts_operating_point(data_dir, qsts_output, args.time, network)
    solver = OpenDSSSolver(data_dir / "network.dss")
    solver.build(network)
    result = run_rms(solver, network, operating_point, rms_config, args.scenario)
    output_dir = args.output.resolve() if args.output else qsts_output / "rms" / args.scenario
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output_dir / "rms_summary.csv", index=False)
    result.bus_voltages.to_csv(output_dir / "rms_bus_voltages.csv", index=False)
    (output_dir / "rms_initialization.json").write_text(json.dumps({
        "qsts_time": operating_point.time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": args.scenario,
        "initial_der_pq": {name: {"p_kw": command.p_kw, "q_kvar": command.q_kvar} for name, command in operating_point.der_pq.items()},
        "initial_der_states": operating_point.der_states,
    }, indent=2), encoding="utf-8")
    print(f"RMS completed: scenario={args.scenario}, qsts_time={args.time}")
    print(f"Results: {output_dir}")
    print(result.summary[["time_s", "event", "min_v_pu", "max_v_pu"]].to_string(index=False))


if __name__ == "__main__":
    main()
