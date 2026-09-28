"""Run any generated QSTS case selected explicitly with --data-dir."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.der_model import BESSModel, DERModel, EVModel, PVModel, WindModel
from src.power_flow import NetworkModel
from src.qsts import run_qsts
from src.solvers.opendss_solver import OpenDSSSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a generated OpenDSS QSTS case")
    parser.add_argument("--data-dir", type=Path, required=True, help="Case directory containing network.json and network.dss.")
    parser.add_argument("--output", type=Path, help="Result directory; defaults to output/<case-directory-name>.")
    return parser.parse_args()


def build_der_models(network: NetworkModel, data_dir: Path) -> list[DERModel]:
    models: list[DERModel] = []
    model_types = {"pv": PVModel, "wind": WindModel, "ev": EVModel}
    for device in network.devices:
        kind = str(device["kind"])
        if kind == "load":
            continue
        profile_file = data_dir / str(device["profile_file"])
        common = {"name": str(device["id"]), "profile_file": profile_file, "bus_name": str(device["bus"])}
        if kind == "bess":
            model: DERModel = BESSModel(
                **common,
                capacity_kwh=float(device["energy_kwh"]),
                reserve_soc=float(device.get("reserve_soc", 20.0)),
                initial_soc=float(device.get("initial_soc", 50.0)),
                charge_efficiency=float(device.get("charge_efficiency", 0.95)),
                discharge_efficiency=float(device.get("discharge_efficiency", 0.95)),
            )
        elif kind in model_types:
            model = model_types[kind](**common)
        else:
            raise ValueError(f"Unsupported DER kind {kind!r} for device {device['id']!r}.")
        if not profile_file.exists():
            raise FileNotFoundError(f"DER profile not found: {profile_file}")
        model.load_profile()
        models.append(model)
    return models


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
    der_models = build_der_models(network, data_dir)

    qsts_bus, qsts_system = run_qsts(solver, der_models, data_dir / "load_profiles.csv")
    qsts_bus.to_csv(output_dir / "qsts_bus_voltages.csv", index=False)
    qsts_system.to_csv(output_dir / "qsts_system.csv", index=False)
    print(f"QSTS completed for case: {data_dir.name}")
    print(f"Input data: {data_dir}")
    print(f"Results: {output_dir.resolve()}")
    print(qsts_system[["hour", "total_load_kw", "min_v_pu", "max_v_pu", "loss_kw"]].to_string(index=False))


if __name__ == "__main__":
    main()
