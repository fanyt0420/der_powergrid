"""Run the generated multi-DER OpenDSS QSTS demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.der_model import BESSModel, EVModel, PVModel, WindModel
from src.opendss_model import OpenDSSModel
from src.qsts import run_qsts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the generated multi-DER QSTS demonstration")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    profiles_dir = data_dir / "der_profiles"
    required_files = [
        data_dir / "feeder.dss", data_dir / "load_profiles.csv",
        profiles_dir / "pv1.csv", profiles_dir / "wind1.csv",
        profiles_dir / "bess1.csv", profiles_dir / "ev1.csv",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Generated input data is missing. Run `python generate_data.py` first.\n"
            + "\n".join(missing)
        )

    args.output.mkdir(parents=True, exist_ok=True)
    model = OpenDSSModel(data_dir / "feeder.dss")
    model.compile()

    for der_model in [
        PVModel("PV1", profiles_dir / "pv1.csv", bus_name="bus1"),
        WindModel("Wind1", profiles_dir / "wind1.csv", bus_name="bus2"),
        BESSModel("BESS1", profiles_dir / "bess1.csv", bus_name="bus2"),
        EVModel("EV1", profiles_dir / "ev1.csv", bus_name="bus3"),
    ]:
        der_model.load_profile()
        model.add_der_model(der_model)

    qsts_bus, qsts_system = run_qsts(model, data_dir / "load_profiles.csv")
    qsts_bus.to_csv(args.output / "qsts_bus_voltages.csv", index=False)
    qsts_system.to_csv(args.output / "qsts_system.csv", index=False)

    print("Multi-DER QSTS completed.")
    print(f"Input data: {data_dir}")
    print(f"Results: {args.output.resolve()}")
    print(qsts_system[["hour", "total_load_kw", "min_v_pu", "max_v_pu", "loss_kw"]].to_string(index=False))


if __name__ == "__main__":
    main()
