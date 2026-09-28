"""Generate all fixed network and time-series inputs for the multi-DER demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

FEEDER = """Clear
Set DefaultBaseFrequency=60
! Generated 12.47-kV three-phase unbalanced feeder for the multi-DER QSTS demo.
New Circuit.MultiDERDemo basekv=12.47 pu=1.0 phases=3 bus1=source
New LineCode.MAIN nphases=3 BaseFreq=60 units=km
~ rmatrix=[0.40 | 0.05 0.40 | 0.05 0.05 0.40]
~ xmatrix=[0.30 | 0.10 0.30 | 0.10 0.10 0.30]
~ cmatrix=[3.0 | 0.0 3.0 | 0.0 0.0 3.0]
New Line.L1 phases=3 bus1=source.1.2.3 bus2=bus1.1.2.3 linecode=MAIN length=1.0 units=km
New Line.L2 phases=3 bus1=bus1.1.2.3 bus2=bus2.1.2.3 linecode=MAIN length=1.2 units=km
New Line.L3 phases=3 bus1=bus2.1.2.3 bus2=bus3.1.2.3 linecode=MAIN length=0.8 units=km
! Load operating P/Q is supplied only by load_profiles.csv.
New Load.Load1 phases=3 bus1=bus1.1.2.3 conn=wye model=1 kV=12.47 kW=0 kvar=0
New Load.Load2 phases=3 bus1=bus2.1.2.3 conn=wye model=1 kV=12.47 kW=0 kvar=0
New Load.Load3 phases=3 bus1=bus3.1.2.3 conn=wye model=1 kV=12.47 kW=0 kvar=0
! DER elements are driven independently by files in der_profiles/.
New Generator.PV1 phases=3 bus1=bus1.1.2.3 kV=12.47 kW=0 kvar=0 model=1
New Generator.Wind1 phases=3 bus1=bus2.1.2.3 kV=12.47 kW=0 kvar=0 model=1
New Storage.BESS1 phases=3 bus1=bus2.1.2.3 kV=12.47 kWrated=100 kWhrated=200
~ %stored=50 %reserve=20 %EffCharge=95 %EffDischarge=95 kW=0 kvar=0 dispmode=DEFAULT
New Load.EV1 phases=3 bus1=bus3.1.2.3 conn=wye model=1 kV=12.47 kW=0 kvar=0
Set VoltageBases=[12.47]
CalcVoltageBases
Set MaxIterations=100
Solve
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-DER demonstration input data")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    return parser.parse_args()


def write_load_profiles(data_dir: Path, hours: list[int]) -> None:
    base_loads = {"Load1": (450.0, 150.0, 1), "Load2": (340.0, 110.0, 3), "Load3": (300.0, 100.0, 5)}
    daily_shape = [0.62, 0.58, 0.55, 0.53, 0.55, 0.62, 0.72, 0.82, 0.88, 0.91, 0.93, 0.95,
                   0.96, 0.97, 0.99, 1.02, 1.07, 1.12, 1.18, 1.22, 1.18, 1.08, 0.90, 0.74]
    rows = []
    for hour in hours:
        for name, (base_kw, base_kvar, shift) in base_loads.items():
            modifier = 1.0 + 0.06 * (((hour + shift) % 5) - 2) / 2
            p_kw = base_kw * daily_shape[hour] * modifier
            rows.append({"hour": hour, "load_name": name, "p_kw": round(p_kw, 3),
                         "q_kvar": round(p_kw * base_kvar / base_kw, 3)})
    pd.DataFrame(rows).to_csv(data_dir / "load_profiles.csv", index=False)


def write_der_profiles(profiles_dir: Path, hours: list[int]) -> None:
    pd.DataFrame({
        "hour": hours,
        "pv_kw": [max(0, 400 * (1 - abs(hour - 12) / 6)) for hour in hours],
        "pv_kvar": 0.0,
        "controller_type": "volt_var",
        "q_limit_kvar": 100.0,
        "v_ref_pu": 1.0,
        "droop_kvar_per_pu": 1000.0,
    }).to_csv(profiles_dir / "pv1.csv", index=False)
    pd.DataFrame({
        "hour": hours,
        "wind_kw": [55 + 25 * ((hour % 6) / 5) for hour in hours],
        "wind_kvar": 0.0,
        "controller_type": "constant_pq",
    }).to_csv(profiles_dir / "wind1.csv", index=False)
    pd.DataFrame({
        "hour": hours,
        "bess_kw": [-25 if 9 <= hour <= 16 else 35 for hour in hours],
        "bess_kvar": 0.0,
        "controller_type": "soc_schedule",
    }).to_csv(profiles_dir / "bess1.csv", index=False)
    pd.DataFrame({
        "hour": hours,
        "ev_kw": [65 if 18 <= hour <= 23 else 0 for hour in hours],
        "ev_kvar": 0.0,
        "controller_type": "constant_pq",
    }).to_csv(profiles_dir / "ev1.csv", index=False)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    profiles_dir = data_dir / "der_profiles"
    data_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "feeder.dss").write_text(FEEDER, encoding="utf-8")
    hours = list(range(24))
    write_load_profiles(data_dir, hours)
    write_der_profiles(profiles_dir, hours)
    print(f"Generated feeder and profiles in: {data_dir}")


if __name__ == "__main__":
    main()
