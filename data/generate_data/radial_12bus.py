"""Generate a synthetic 12-bus, multi-lateral, multi-DER three-phase QSTS case.

This is intentionally not labelled IEEE 123: it is a reproducible, more complex
synthetic feeder for exercising the data and solver interfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from topology_plot import plot_network_topology

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "data" / "radial_12bus"
HOURS = list(range(24))
START = pd.Timestamp("2026-01-01 00:00:00")
STEP_SECONDS = 3600
EDGES = [("source", "b1", 0.20), ("b1", "b2", 0.18), ("b2", "b3", 0.16), ("b3", "b4", 0.14), ("b2", "b5", 0.17), ("b5", "b6", 0.12), ("b5", "b7", 0.15), ("b1", "b8", 0.19), ("b8", "b9", 0.13), ("b9", "b10", 0.12), ("b8", "b11", 0.15), ("b11", "b12", 0.12)]
Z_MATRIX = [[[0.16, 0.12], [0.02, 0.04], [0.02, 0.04]], [[0.02, 0.04], [0.16, 0.12], [0.02, 0.04]], [[0.02, 0.04], [0.02, 0.04], [0.16, 0.12]]]
TOPOLOGY_POSITIONS = {
    "source": (0, 0), "b1": (1, 0), "b2": (2, 0), "b3": (3, 1), "b4": (4, 1),
    "b5": (3, -1), "b6": (4, -1.6), "b7": (4, -0.5), "b8": (2, -3), "b9": (3, -3.6),
    "b10": (4, -3.6), "b11": (3, -2.4), "b12": (4, -2.4),
}


def timestamps() -> list[pd.Timestamp]:
    return [START + pd.Timedelta(seconds=STEP_SECONDS * h) for h in HOURS]


def simulation() -> dict:
    ts = timestamps()
    return {
        "start_datetime": ts[0].strftime("%Y-%m-%d %H:%M:%S"),
        "end_datetime": (ts[-1] + pd.Timedelta(seconds=STEP_SECONDS)).strftime("%Y-%m-%d %H:%M:%S"),
        "step_seconds": STEP_SECONDS,
    }


def rms_config() -> dict:
    """Dynamic parameters only; topology and QSTS setpoints remain in existing files."""
    return {
        "nominal_frequency_hz": 60.0,
        "dynamic_devices": [
            {"device_id": "PV1", "model_type": "gfl_inverter", "current_limit_pu": 1.25, "pll_kp_hz_per_rad": 12.0, "pll_ki_hz_per_rad_s": 180.0, "tau_p_control_s": 0.08, "tau_q_control_s": 0.06, "tau_current_s": 0.015},
            {"device_id": "PV2", "model_type": "gfm_inverter", "current_limit_pu": 1.25, "inertia_s": 1.5, "damping_pu_per_hz": 0.25, "p_droop_pu_per_hz": 0.08, "q_droop_pu_per_pu": 2.0, "voltage_droop_pu_per_pu": 0.5, "tau_power_control_s": 0.08, "tau_power_measure_s": 0.04, "tau_voltage_control_s": 0.05},
            {"device_id": "Wind1", "model_type": "aggregate_der", "current_limit_pu": 1.20, "tau_p_control_s": 0.25, "tau_q_control_s": 0.15, "tau_measure_s": 0.05, "voltage_support_pu_per_pu": 1.0},
            {"device_id": "BESS1", "model_type": "gfl_inverter", "current_limit_pu": 1.30, "pll_kp_hz_per_rad": 10.0, "pll_ki_hz_per_rad_s": 150.0, "tau_p_control_s": 0.10, "tau_q_control_s": 0.08, "tau_current_s": 0.02},
            {"device_id": "EV1", "model_type": "aggregate_der", "current_limit_pu": 1.10, "tau_p_control_s": 0.30, "tau_q_control_s": 0.20, "tau_measure_s": 0.10, "voltage_support_pu_per_pu": 0.0},
        ],
        "scenarios": {
            "flat_run": {"dt_s": 0.02, "t_end_s": 2.0, "events": []},
            "load_step": {"dt_s": 0.02, "t_end_s": 2.0, "events": [{"time_s": 1.0, "type": "load_scale", "device_id": "Load12", "p_multiplier": 1.10, "q_multiplier": 1.10}]},
            "der_trip": {"dt_s": 0.02, "t_end_s": 2.0, "events": [{"time_s": 1.0, "type": "der_trip", "device_id": "PV1"}]},
        },
    }


def network() -> dict:
    devices = [{"id": f"Load{i}", "kind": "load", "bus": f"b{i}", "phases": [1, 2, 3], "connection": "wye", "kv_ll": 12.47} for i in range(1, 13)]
    devices += [
        {"id": "PV1", "kind": "pv", "bus": "b4", "phases": [1, 2, 3], "p_rated_kw": 180, "profile_file": "der_profiles/pv1.csv"},
        {"id": "PV2", "kind": "pv", "bus": "b10", "phases": [1, 2, 3], "p_rated_kw": 120, "profile_file": "der_profiles/pv2.csv"},
        {"id": "Wind1", "kind": "wind", "bus": "b7", "phases": [1, 2, 3], "p_rated_kw": 100, "profile_file": "der_profiles/wind1.csv"},
        {"id": "BESS1", "kind": "bess", "bus": "b6", "phases": [1, 2, 3], "p_rated_kw": 80, "energy_kwh": 500, "reserve_soc": 20, "initial_soc": 55, "profile_file": "der_profiles/bess1.csv"},
        {"id": "EV1", "kind": "ev", "bus": "b12", "phases": [1, 2, 3], "p_rated_kw": 90, "profile_file": "der_profiles/ev1.csv"},
    ]
    return {"base": {"frequency_hz": 60, "base_kv_ll": 12.47, "slack_bus": "source"}, "simulation": simulation(), "buses": [{"id": "source", "phases": [1, 2, 3], "is_slack": True}] + [{"id": f"b{i}", "phases": [1, 2, 3]} for i in range(1, 13)], "branches": [{"id": f"L{i}", "from_bus": start, "to_bus": end, "phases": [1, 2, 3], "length_km": length, "z_ohm_per_km": Z_MATRIX} for i, (start, end, length) in enumerate(EDGES, 1)], "devices": devices}


def dss_text() -> str:
    lines = ["Clear", "Set DefaultBaseFrequency=60", "New Circuit.Radial12Bus basekv=12.47 pu=1.0 phases=3 bus1=source", "New LineCode.MAIN nphases=3 BaseFreq=60 units=km", "~ rmatrix=[0.16 | 0.02 0.16 | 0.02 0.02 0.16]", "~ xmatrix=[0.12 | 0.04 0.12 | 0.04 0.04 0.12]"]
    lines += [f"New Line.L{i} phases=3 bus1={start}.1.2.3 bus2={end}.1.2.3 linecode=MAIN length={length} units=km" for i, (start, end, length) in enumerate(EDGES, 1)]
    lines += [f"New Load.Load{i} phases=3 bus1=b{i}.1.2.3 conn=wye model=1 kV=12.47 kW=0 kvar=0" for i in range(1, 13)]
    lines += ["New Generator.PV1 phases=3 bus1=b4.1.2.3 kV=12.47 kW=0 kvar=0 model=1", "New Generator.PV2 phases=3 bus1=b10.1.2.3 kV=12.47 kW=0 kvar=0 model=1", "New Generator.Wind1 phases=3 bus1=b7.1.2.3 kV=12.47 kW=0 kvar=0 model=1", "New Storage.BESS1 phases=3 bus1=b6.1.2.3 kV=12.47 kWrated=80 kWhrated=500 %stored=55 %reserve=20 %EffCharge=95 %EffDischarge=95 kW=0 kvar=0 dispmode=DEFAULT", "New Load.EV1 phases=3 bus1=b12.1.2.3 conn=wye model=1 kV=12.47 kW=0 kvar=0", "Set VoltageBases=[12.47]", "CalcVoltageBases", "Set MaxIterations=100", "Solve"]
    return "\n".join(lines) + "\n"


def main() -> None:
    profiles = CASE_DIR / "der_profiles"
    CASE_DIR.mkdir(parents=True, exist_ok=True); profiles.mkdir(exist_ok=True)
    case_network = network()
    (CASE_DIR / "network.json").write_text(json.dumps(case_network, indent=2), encoding="utf-8")
    (CASE_DIR / "network.dss").write_text(dss_text(), encoding="utf-8")
    (CASE_DIR / "rms_config.json").write_text(json.dumps(rms_config(), indent=2), encoding="utf-8")
    plot_network_topology(case_network, CASE_DIR / "network_topology.png", TOPOLOGY_POSITIONS)
    ts_text = [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps()]
    shape = [0.55, 0.52, 0.50, 0.49, 0.50, 0.57, 0.68, 0.79, 0.87, 0.92, 0.96, 0.98, 1.0, 1.01, 1.03, 1.06, 1.10, 1.15, 1.20, 1.22, 1.16, 1.04, 0.84, 0.68]
    rows = []
    for hour in HOURS:
        for index in range(1, 13):
            base_kw, pf = 55 + 8 * (index % 6), 0.94 + 0.01 * (index % 3)
            p_kw = base_kw * shape[hour] * (1 + 0.08 * (((hour + index) % 5) - 2) / 2)
            rows.append({"timestamp": ts_text[hour], "load_name": f"Load{index}", "p_kw": round(p_kw, 3), "q_kvar": round(p_kw * (1 / pf**2 - 1) ** 0.5, 3)})
    pd.DataFrame(rows).to_csv(CASE_DIR / "load_profiles.csv", index=False)
    def pv_file(name: str, rating: float, shift: int) -> None:
        pd.DataFrame({"timestamp": ts_text, "pv_kw": [max(0, rating * (1 - abs(h - (12 + shift)) / 6)) for h in HOURS], "pv_kvar": 0.0, "controller_type": "volt_var", "q_limit_kvar": rating * 0.25, "v_ref_pu": 1.0, "droop_kvar_per_pu": rating * 3}).to_csv(profiles / name, index=False)
    pv_file("pv1.csv", 180, 0); pv_file("pv2.csv", 120, 1)
    pd.DataFrame({"timestamp": ts_text, "wind_kw": [65 + 35 * ((h + 2) % 6) / 5 for h in HOURS], "wind_kvar": 0.0, "controller_type": "constant_pq"}).to_csv(profiles / "wind1.csv", index=False)
    pd.DataFrame({"timestamp": ts_text, "bess_kw": [-45 if 10 <= h <= 15 else 35 for h in HOURS], "bess_kvar": 0.0, "controller_type": "soc_schedule"}).to_csv(profiles / "bess1.csv", index=False)
    pd.DataFrame({"timestamp": ts_text, "ev_kw": [90 if 18 <= h <= 22 else 0 for h in HOURS], "ev_kvar": 0.0, "controller_type": "constant_pq"}).to_csv(profiles / "ev1.csv", index=False)
    print(f"Generated synthetic radial_12bus case in: {CASE_DIR}")


if __name__ == "__main__":
    main()
