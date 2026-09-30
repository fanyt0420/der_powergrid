"""Generate a complete solver-neutral IEEE 13-node feeder with DER profiles."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from topology_plot import plot_network_topology


ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "data" / "ieee13_unbalanced_der"
BASE = CASE / "ieee13_base"
URL = "https://raw.githubusercontent.com/dss-extensions/electricdss-tst/master/Version8/Distrib/IEEETestCases"
HOURS = list(range(24))

BUS = {
    "SourceBus": [1, 2, 3], "650": [1, 2, 3], "RG60": [1, 2, 3], "632": [1, 2, 3],
    "633": [1, 2, 3], "634": [1, 2, 3], "645": [2, 3], "646": [2, 3],
    "670": [1, 2, 3], "671": [1, 2, 3], "680": [1, 2, 3], "692": [1, 2, 3],
    "675": [1, 2, 3], "684": [1, 3], "611": [3], "652": [1],
}
POS = {
    "SourceBus": (-1, 0), "650": (0, 0), "RG60": (1, 0), "632": (2, 0), "670": (3, 0),
    "671": (4, 0), "680": (5, 0), "633": (3, 2), "634": (4, 2), "645": (3, -2),
    "646": (4, -2), "692": (5, -1), "675": (6, -1), "684": (5, 2), "611": (6, 2), "652": (6, 3),
}

# Matrices are the exact local IEEE13 line-code values in ohm/mile and nF/mile.
LINE_CODES = {
    "mtx601": {"unit": "mi", "r_ohm_per_unit": [[.3465, .1560, .1580], [.1560, .3375, .1535], [.1580, .1535, .3414]], "x_ohm_per_unit": [[1.0179, .5017, .4236], [.5017, 1.0478, .3849], [.4236, .3849, 1.0348]]},
    "mtx602": {"unit": "mi", "r_ohm_per_unit": [[.7526, .1580, .1560], [.1580, .7475, .1535], [.1560, .1535, .7436]], "x_ohm_per_unit": [[1.1814, .4236, .5017], [.4236, 1.1983, .3849], [.5017, .3849, 1.2112]]},
    "mtx603": {"unit": "mi", "r_ohm_per_unit": [[1.3238, .2066], [.2066, 1.3294]], "x_ohm_per_unit": [[1.3569, .4591], [.4591, 1.3471]]},
    "mtx604": {"unit": "mi", "r_ohm_per_unit": [[1.3238, .2066], [.2066, 1.3294]], "x_ohm_per_unit": [[1.3569, .4591], [.4591, 1.3471]]},
    "mtx605": {"unit": "mi", "r_ohm_per_unit": [[1.3292]], "x_ohm_per_unit": [[1.3475]]},
    "mtx606": {"unit": "mi", "r_ohm_per_unit": [[.791721, .318476, .28345], [.318476, .781649, .318476], [.28345, .318476, .791721]], "x_ohm_per_unit": [[.438352, .0276838, -.0184204], [.0276838, .396697, .0276838], [-.0184204, .0276838, .438352]], "c_nf_per_unit": [[383.948, 0, 0], [0, 383.948, 0], [0, 0, 383.948]]},
    "mtx607": {"unit": "mi", "r_ohm_per_unit": [[1.3425]], "x_ohm_per_unit": [[.5124]], "c_nf_per_unit": [[236.0]]},
}
LINE_NORMAL_AMPS = {"mtx601": 600.0, "mtx602": 300.0, "mtx603": 250.0, "mtx604": 250.0, "mtx605": 150.0, "mtx606": 300.0, "mtx607": 150.0}

LINE_DATA = [
    ("650632", "RG60", [1, 2, 3], "632", [1, 2, 3], "mtx601", 2000),
    ("632670", "632", [1, 2, 3], "670", [1, 2, 3], "mtx601", 667),
    ("670671", "670", [1, 2, 3], "671", [1, 2, 3], "mtx601", 1333),
    ("671680", "671", [1, 2, 3], "680", [1, 2, 3], "mtx601", 1000),
    ("632633", "632", [1, 2, 3], "633", [1, 2, 3], "mtx602", 500),
    ("632645", "632", [3, 2], "645", [3, 2], "mtx603", 500),
    ("645646", "645", [3, 2], "646", [3, 2], "mtx603", 300),
    ("692675", "692", [1, 2, 3], "675", [1, 2, 3], "mtx606", 500),
    ("671684", "671", [1, 3], "684", [1, 3], "mtx604", 300),
    ("684611", "684", [3], "611", [3], "mtx605", 300),
    ("684652", "684", [1], "652", [1], "mtx607", 800),
]

LOADS = [
    ("671", "671", [1, 2, 3], "delta", 1, 4.16), ("634a", "634", [1], "wye", 1, .277),
    ("634b", "634", [2], "wye", 1, .277), ("634c", "634", [3], "wye", 1, .277),
    ("645", "645", [2], "wye", 1, 2.4), ("646", "646", [2, 3], "delta", 2, 4.16),
    ("692", "692", [3, 1], "delta", 5, 4.16), ("675a", "675", [1], "wye", 1, 2.4),
    ("675b", "675", [2], "wye", 1, 2.4), ("675c", "675", [3], "wye", 1, 2.4),
    ("611", "611", [3], "wye", 5, 2.4), ("652", "652", [1], "wye", 2, 2.4),
    ("670a", "670", [1], "wye", 1, 2.4), ("670b", "670", [2], "wye", 1, 2.4),
    ("670c", "670", [3], "wye", 1, 2.4),
]


def _line_branch(entry: tuple) -> dict:
    ident, from_bus, from_nodes, to_bus, to_nodes, code, length_ft = entry
    return {"id": ident, "kind": "line", "from_bus": from_bus, "to_bus": to_bus, "from_terminal_nodes": from_nodes, "to_terminal_nodes": to_nodes, "phases": from_nodes, "line_code": code, "length": length_ft, "length_unit": "ft", "length_km": length_ft * .0003048, "normal_amps": LINE_NORMAL_AMPS[code], "rating_source": "engineering_screening_assumption", "in_service": True}


def _transformer_branches() -> list[dict]:
    sub = {"id": "Sub", "kind": "transformer", "from_bus": "SourceBus", "to_bus": "650", "phases": [1, 2, 3], "xhl_percent": .008, "windings": [{"bus": "SourceBus", "terminal_nodes": [1, 2, 3], "connection": "delta", "kv_ll": 115.0, "kva": 5000.0, "resistance_percent": .0005}, {"bus": "650", "terminal_nodes": [1, 2, 3], "connection": "wye", "kv_ll": 4.16, "kva": 5000.0, "resistance_percent": .0005}], "in_service": True}
    regulators = []
    for phase in (1, 2, 3):
        regulators.append({"id": f"Reg{phase}", "kind": "regulator", "from_bus": "650", "to_bus": "RG60", "phases": [phase], "xhl_percent": .01, "load_loss_percent": .01, "windings": [{"bus": "650", "terminal_nodes": [phase], "connection": "wye", "kv_ln": 2.4, "kva": 1666.0}, {"bus": "RG60", "terminal_nodes": [phase], "connection": "wye", "kv_ln": 2.4, "kva": 1666.0}], "tap": {"position": 0, "min_pu": .9, "max_pu": 1.1, "num_taps": 32}, "control": {"type": "voltage_regulator", "controlled_winding": 2, "vreg_v": 122.0, "band_v": 2.0, "ptratio": 20.0, "ctprim_a": 700.0, "r_v": 3.0, "x_v": 9.0}, "in_service": True})
    xfm1 = {"id": "XFM1", "kind": "transformer", "from_bus": "633", "to_bus": "634", "phases": [1, 2, 3], "xhl_percent": 2.0, "windings": [{"bus": "633", "terminal_nodes": [1, 2, 3], "connection": "wye", "kv_ll": 4.16, "kva": 500.0, "resistance_percent": .55}, {"bus": "634", "terminal_nodes": [1, 2, 3], "connection": "wye", "kv_ll": .48, "kva": 500.0, "resistance_percent": .55}], "in_service": True}
    return [sub, *regulators, xfm1]


def _network(ders: list[dict]) -> dict:
    nominal = {bus: 115.0 if bus == "SourceBus" else .48 if bus == "634" else 4.16 for bus in BUS}
    load_devices = [{"id": ident, "kind": "load", "bus": bus, "phases": nodes, "terminal_nodes": nodes, "connection": connection, "load_model": model, "kv": kv, "base_p_kw": p, "base_q_kvar": q} for (ident, bus, nodes, connection, model, kv), (p, q) in zip(LOADS, _base_loads(), strict=True)]
    switch = {"id": "671692", "kind": "switch", "from_bus": "671", "to_bus": "692", "from_terminal_nodes": [1, 2, 3], "to_terminal_nodes": [1, 2, 3], "phases": [1, 2, 3], "normally_closed": True, "r1_ohm": 1e-4, "r0_ohm": 1e-4, "x1_ohm": 0.0, "x0_ohm": 0.0, "normal_amps": 600.0, "rating_source": "engineering_screening_assumption", "in_service": True}
    return {"schema_version": "1.0", "base": {"frequency_hz": 60, "base_kv_ll": 4.16, "slack_bus": "SourceBus"}, "constraints": {"voltage_min_pu": .95, "voltage_max_pu": 1.05, "voltage_unbalance_limit_pct": 2.0, "line_loading_limit_pct": 100.0, "transformer_loading_limit_pct": 100.0, "reverse_power_tolerance_kw": 1e-6}, "source": {"id": "Source", "bus": "SourceBus", "phases": [1, 2, 3], "base_kv_ll": 115.0, "voltage_pu": 1.0001, "angle_deg": 30.0, "mvasc3": 20000.0, "mvasc1": 21000.0}, "simulation": {"start_datetime": "2026-01-01 00:00:00", "end_datetime": "2026-01-02 00:00:00", "step_seconds": 3600}, "buses": [{"id": bus, "phases": phases, "nominal_kv_ll": nominal[bus], "is_slack": bus == "SourceBus"} for bus, phases in BUS.items()], "line_codes": LINE_CODES, "branches": [*_transformer_branches(), *[_line_branch(entry) for entry in LINE_DATA], switch], "shunt_devices": [{"id": "Cap1", "kind": "capacitor", "bus": "675", "terminal_nodes": [1, 2, 3], "phases": [1, 2, 3], "connection": "wye", "kv_ll": 4.16, "q_kvar": 600.0, "in_service": True}, {"id": "Cap2", "kind": "capacitor", "bus": "611", "terminal_nodes": [3], "phases": [3], "connection": "wye", "kv_ln": 2.4, "q_kvar": 100.0, "in_service": True}], "devices": [*load_devices, *ders]}


def _base_loads() -> list[tuple[float, float]]:
    values = {"671": (1155, 660), "634a": (160, 110), "634b": (120, 90), "634c": (120, 90), "645": (170, 125), "646": (230, 132), "692": (170, 151), "675a": (485, 190), "675b": (68, 60), "675c": (290, 212), "611": (170, 80), "652": (128, 86), "670a": (17, 10), "670b": (66, 38), "670c": (117, 68)}
    return [values[ident] for ident, *_ in LOADS]


def main() -> None:
    CASE.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(exist_ok=True)
    profiles = CASE / "der_profiles"
    profiles.mkdir(exist_ok=True)
    master, codes = BASE / "IEEE13Nodeckt.dss", BASE / "IEEELineCodes.DSS"
    if not master.exists():
        urlretrieve(URL + "/13Bus/IEEE13Nodeckt.dss", master)
    if not codes.exists():
        urlretrieve(URL + "/IEEELineCodes.DSS", codes)
    master.write_text(master.read_text(encoding="utf-8").replace("BusCoords IEEE13Node_BusXY.csv", "! Bus coordinates omitted; see network_topology.png"), encoding="utf-8")

    ders = [
        {"id": "PV_632_A", "kind": "pv", "bus": "632", "phases": [1], "terminal_nodes": [1], "connection": "wye", "kv": 2.4, "p_rated_kw": 70}, {"id": "PV_670_C", "kind": "pv", "bus": "670", "phases": [3], "terminal_nodes": [3], "connection": "wye", "kv": 2.4, "p_rated_kw": 50},
        {"id": "PV_671_A", "kind": "pv", "bus": "671", "phases": [1], "terminal_nodes": [1], "connection": "wye", "kv": 2.4, "p_rated_kw": 80}, {"id": "PV_675_A", "kind": "pv", "bus": "675", "phases": [1], "terminal_nodes": [1], "connection": "wye", "kv": 2.4, "p_rated_kw": 100},
        {"id": "PV_675_C", "kind": "pv", "bus": "675", "phases": [3], "terminal_nodes": [3], "connection": "wye", "kv": 2.4, "p_rated_kw": 70}, {"id": "PV_634_ABC", "kind": "pv", "bus": "634", "phases": [1, 2, 3], "terminal_nodes": [1, 2, 3], "connection": "wye", "kv": .48, "p_rated_kw": 180},
        {"id": "BESS_671_B", "kind": "bess", "bus": "671", "phases": [2], "terminal_nodes": [2], "connection": "wye", "kv": 2.4, "p_rated_kw": 100, "energy_kwh": 400, "reserve_soc": 20, "initial_soc": 55}, {"id": "BESS_645_B", "kind": "bess", "bus": "645", "phases": [2], "terminal_nodes": [2], "connection": "wye", "kv": 2.4, "p_rated_kw": 60, "energy_kwh": 240, "reserve_soc": 20, "initial_soc": 60},
        {"id": "BESS_684_A", "kind": "bess", "bus": "684", "phases": [1], "terminal_nodes": [1], "connection": "wye", "kv": 2.4, "p_rated_kw": 50, "energy_kwh": 200, "reserve_soc": 20, "initial_soc": 50}, {"id": "EV_671_C", "kind": "ev", "bus": "671", "phases": [3], "terminal_nodes": [3], "connection": "wye", "kv": 2.4, "p_rated_kw": 60},
        {"id": "EV_675_B", "kind": "ev", "bus": "675", "phases": [2], "terminal_nodes": [2], "connection": "wye", "kv": 2.4, "p_rated_kw": 50}, {"id": "EV_611_C", "kind": "ev", "bus": "611", "phases": [3], "terminal_nodes": [3], "connection": "wye", "kv": 2.4, "p_rated_kw": 40},
        {"id": "Wind_680_ABC", "kind": "wind", "bus": "680", "phases": [1, 2, 3], "terminal_nodes": [1, 2, 3], "connection": "wye", "kv": 2.4, "p_rated_kw": 120},
    ]
    for device in ders:
        device["profile_file"] = f"der_profiles/{device['id'].lower()}.csv"
    network = _network(ders)
    (CASE / "network.json").write_text(json.dumps(network, indent=2), encoding="utf-8")

    dss = [f'Redirect "{master.resolve()}"']
    for device in ders:
        phases, count, kv = ".".join(map(str, device["terminal_nodes"])), len(device["terminal_nodes"]), device["kv"]
        if device["kind"] in {"pv", "wind"}:
            dss.append(f"New Generator.{device['id']} phases={count} bus1={device['bus']}.{phases} kV={kv} kW=0 kvar=0")
        elif device["kind"] == "ev":
            dss.append(f"New Load.{device['id']} phases={count} bus1={device['bus']}.{phases} conn=wye kV={kv} kW=0 kvar=0")
        else:
            dss.append(f"New Storage.{device['id']} phases={count} bus1={device['bus']}.{phases} kV={kv} kWrated={device['p_rated_kw']} kWhrated={device['energy_kwh']} %stored={device['initial_soc']} %reserve={device['reserve_soc']} dispmode=EXTERNAL kW=0 kvar=0")
    (CASE / "network.dss").write_text("\n".join([*dss, "Solve", ""]), encoding="utf-8")

    shape = [.55, .52, .5, .5, .53, .62, .72, .82, .9, .96, 1, 1.02, 1, 1.01, 1.05, 1.1, 1.18, 1.25, 1.3, 1.28, 1.15, .95, .75, .65]
    rows = []
    for hour in HOURS:
        for (ident, *_), (p_kw, q_kvar) in zip(LOADS, _base_loads(), strict=True):
            factor = shape[hour] * (1 + {"675a": .08, "675b": -.05, "675c": .03}.get(ident, 0))
            rows.append({"timestamp": f"2026-01-01 {hour:02d}:00:00", "load_name": ident, "p_kw": round(p_kw * factor, 3), "q_kvar": round(q_kvar * factor, 3)})
    pd.DataFrame(rows).to_csv(CASE / "load_profiles.csv", index=False)
    timestamps = [f"2026-01-01 {hour:02d}:00:00" for hour in HOURS]
    for index, device in enumerate(ders):
        file, rating = profiles / f"{device['id'].lower()}.csv", device["p_rated_kw"]
        if device["kind"] == "pv":
            pd.DataFrame({"timestamp": timestamps, "pv_kw": [max(0, rating * (1 - abs(hour - (12 + index % 3)) / 6)) for hour in HOURS], "pv_kvar": 0, "controller_type": "volt_var", "q_limit_kvar": rating * .35, "v_ref_pu": 1, "droop_kvar_per_pu": rating * 5}).to_csv(file, index=False)
        elif device["kind"] == "wind":
            pd.DataFrame({"timestamp": timestamps, "wind_kw": [rating * (.45 + .45 * ((hour + index) % 6) / 5) for hour in HOURS], "wind_kvar": 0, "controller_type": "constant_pq"}).to_csv(file, index=False)
        elif device["kind"] == "bess":
            pd.DataFrame({"timestamp": timestamps, "bess_kw": [-.6 * rating if 10 <= hour <= 15 else .5 * rating if 18 <= hour <= 21 else 0 for hour in HOURS], "bess_kvar": 0, "controller_type": "soc_schedule"}).to_csv(file, index=False)
        else:
            pd.DataFrame({"timestamp": timestamps, "ev_kw": [rating if 18 <= hour <= 22 else 0 for hour in HOURS], "ev_kvar": 0, "controller_type": "constant_pq"}).to_csv(file, index=False)
    plot_network_topology(network, CASE / "network_topology.png", POS)
    print(CASE)


if __name__ == "__main__":
    main()
