"""Generate the original three-load, four-DER demonstration case."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from topology_plot import plot_network_topology

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "data" / "three_node"
HOURS = list(range(24))
Z_MATRIX = [[[0.40, 0.30], [0.05, 0.10], [0.05, 0.10]], [[0.05, 0.10], [0.40, 0.30], [0.05, 0.10]], [[0.05, 0.10], [0.05, 0.10], [0.40, 0.30]]]
TOPOLOGY_POSITIONS = {"source": (0, 0), "bus1": (2, 0), "bus2": (4, 0), "bus3": (6, 0)}


def network() -> dict:
    devices = [
        {"id": f"Load{i}", "kind": "load", "bus": f"bus{i}", "phases": [1, 2, 3], "connection": "wye", "kv_ll": 12.47}
        for i in range(1, 4)
    ]
    devices += [
        {"id": "PV1", "kind": "pv", "bus": "bus1", "phases": [1, 2, 3], "p_rated_kw": 400, "profile_file": "der_profiles/pv1.csv"},
        {"id": "Wind1", "kind": "wind", "bus": "bus2", "phases": [1, 2, 3], "p_rated_kw": 80, "profile_file": "der_profiles/wind1.csv"},
        {"id": "BESS1", "kind": "bess", "bus": "bus2", "phases": [1, 2, 3], "p_rated_kw": 100, "energy_kwh": 200, "reserve_soc": 20, "initial_soc": 50, "profile_file": "der_profiles/bess1.csv"},
        {"id": "EV1", "kind": "ev", "bus": "bus3", "phases": [1, 2, 3], "p_rated_kw": 65, "profile_file": "der_profiles/ev1.csv"},
    ]
    return {"base": {"frequency_hz": 60, "base_kv_ll": 12.47, "slack_bus": "source"}, "buses": [{"id": "source", "phases": [1, 2, 3], "is_slack": True}] + [{"id": f"bus{i}", "phases": [1, 2, 3]} for i in range(1, 4)], "branches": [{"id": f"L{i}", "from_bus": "source" if i == 1 else f"bus{i - 1}", "to_bus": f"bus{i}", "phases": [1, 2, 3], "length_km": length, "z_ohm_per_km": Z_MATRIX} for i, length in enumerate([1.0, 1.2, 0.8], 1)], "devices": devices}


def dss_text() -> str:
    lines = ["Clear", "Set DefaultBaseFrequency=60", "New Circuit.ThreeNodeDemo basekv=12.47 pu=1.0 phases=3 bus1=source", "New LineCode.MAIN nphases=3 BaseFreq=60 units=km", "~ rmatrix=[0.40 | 0.05 0.40 | 0.05 0.05 0.40]", "~ xmatrix=[0.30 | 0.10 0.30 | 0.10 0.10 0.30]"]
    for index, length in enumerate([1.0, 1.2, 0.8], 1):
        start = "source" if index == 1 else f"bus{index - 1}"
        lines.append(f"New Line.L{index} phases=3 bus1={start}.1.2.3 bus2=bus{index}.1.2.3 linecode=MAIN length={length} units=km")
    lines += [f"New Load.Load{i} phases=3 bus1=bus{i}.1.2.3 conn=wye model=1 kV=12.47 kW=0 kvar=0" for i in range(1, 4)]
    lines += ["New Generator.PV1 phases=3 bus1=bus1.1.2.3 kV=12.47 kW=0 kvar=0 model=1", "New Generator.Wind1 phases=3 bus1=bus2.1.2.3 kV=12.47 kW=0 kvar=0 model=1", "New Storage.BESS1 phases=3 bus1=bus2.1.2.3 kV=12.47 kWrated=100 kWhrated=200 %stored=50 %reserve=20 %EffCharge=95 %EffDischarge=95 kW=0 kvar=0 dispmode=DEFAULT", "New Load.EV1 phases=3 bus1=bus3.1.2.3 conn=wye model=1 kV=12.47 kW=0 kvar=0", "Set VoltageBases=[12.47]", "CalcVoltageBases", "Set MaxIterations=100", "Solve"]
    return "\n".join(lines) + "\n"


def main() -> None:
    profiles = CASE_DIR / "der_profiles"
    CASE_DIR.mkdir(parents=True, exist_ok=True); profiles.mkdir(exist_ok=True)
    case_network = network()
    (CASE_DIR / "network.json").write_text(json.dumps(case_network, indent=2), encoding="utf-8")
    (CASE_DIR / "network.dss").write_text(dss_text(), encoding="utf-8")
    plot_network_topology(case_network, CASE_DIR / "network_topology.png", TOPOLOGY_POSITIONS)
    shape = [0.62, 0.58, 0.55, 0.53, 0.55, 0.62, 0.72, 0.82, 0.88, 0.91, 0.93, 0.95, 0.96, 0.97, 0.99, 1.02, 1.07, 1.12, 1.18, 1.22, 1.18, 1.08, 0.90, 0.74]
    rows = []
    for hour in HOURS:
        for name, base_kw, pf, shift in [("Load1", 450, 0.95, 1), ("Load2", 340, 0.95, 3), ("Load3", 300, 0.95, 5)]:
            p_kw = base_kw * shape[hour] * (1 + 0.06 * (((hour + shift) % 5) - 2) / 2)
            rows.append({"hour": hour, "load_name": name, "p_kw": round(p_kw, 3), "q_kvar": round(p_kw * (1 / pf**2 - 1) ** 0.5, 3)})
    pd.DataFrame(rows).to_csv(CASE_DIR / "load_profiles.csv", index=False)
    pd.DataFrame({"hour": HOURS, "pv_kw": [max(0, 400 * (1 - abs(h - 12) / 6)) for h in HOURS], "pv_kvar": 0.0, "controller_type": "volt_var", "q_limit_kvar": 100.0, "v_ref_pu": 1.0, "droop_kvar_per_pu": 1000.0}).to_csv(profiles / "pv1.csv", index=False)
    pd.DataFrame({"hour": HOURS, "wind_kw": [55 + 25 * ((h % 6) / 5) for h in HOURS], "wind_kvar": 0.0, "controller_type": "constant_pq"}).to_csv(profiles / "wind1.csv", index=False)
    pd.DataFrame({"hour": HOURS, "bess_kw": [-25 if 9 <= h <= 16 else 35 for h in HOURS], "bess_kvar": 0.0, "controller_type": "soc_schedule"}).to_csv(profiles / "bess1.csv", index=False)
    pd.DataFrame({"hour": HOURS, "ev_kw": [65 if 18 <= h <= 23 else 0 for h in HOURS], "ev_kvar": 0.0, "controller_type": "constant_pq"}).to_csv(profiles / "ev1.csv", index=False)
    print(f"Generated three_node case in: {CASE_DIR}")


if __name__ == "__main__":
    main()
