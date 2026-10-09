"""Build an IEEE-13 case in which *every* time series comes from a measured dataset.

Self-contained
--------------
This script does NOT read ``data/ieee13_unbalanced_der`` and does NOT import any
other generator. It carries its own IEEE-13 topology, DER fleet, DSS stubs and
RMS configuration, so the case can be regenerated even if the baseline case
directory is absent. Running the case (``run.py`` / ``run_rms.py``) has always
been independent; only *generation* used to depend on the baseline.

Data sources
------------
  PV    <- DDRE-33 node_33 (Chen et al., *Scientific Data* 2025,
           DOI 10.1038/s41597-025-06464-w)
  wind  <- DDRE-33 node_22 (same DOI)
  load  <- Ausgrid "Solar home electricity data", Customer 12, 30-minute
           resolution (Ratnam et al. 2017, DOI 10.1080/14786451.2015.1100196)
  EV    <- Residential charging sessions, 12 sites in Norway, 35k sessions
           (Sorensen et al. 2024, DOI 10.1016/j.dib.2024.110883)
  BESS  <- rule-based dispatch on the measured PV trace (the only non-measured item)

Energy-preserving load scaling
------------------------------
Each feeder load keeps its *own* daily mean power: the Ausgrid curve is rescaled
so its daily mean equals the daily mean of the IEEE-13 load it replaces. Total
daily energy per load is therefore preserved and the QSTS operating point stays
comparable with the baseline; only the intra-day shape becomes real. Reactive
power follows that load's own Q/P ratio.

Known limitations (also written to profile_meta.json)
-----------------------------------------------------
1. The three data sources are not co-located and not observed on the same day.
   This is a *representative* day assembled from independent measurements, not a
   jointly observed one.
2. Ausgrid provides one customer, so the 15 feeder loads reuse that customer on
   15 different dates (diversity comes from date-to-date variation, not from
   household-to-household variation).
3. Season mapping: the DDRE-33 day is 2022-09-20 (northern autumn), so Ausgrid
   dates are drawn from 2012-03/04/05 (southern autumn, same solar geometry).
4. EV charging is modelled as constant power until the session energy is
   delivered. Real sessions taper near full SOC (CC-CV), which is not modelled.
5. All PV nodes share one measured trace, so their pairwise correlation is 1.
   A sub-km cluster is physically plausible, but cloud-level decorrelation is
   not represented here.

Run:  venv/Scripts/python.exe data/generate_data/ieee13_measured.py

Resolution
----------
The simulation step is a run-time choice, independent of the datasets:

    --step-seconds 900     # default, 96 points/day
    --step-seconds 1800    # 48 points/day
    --step-seconds 3600    # 24 points/day (matches the baseline case)
    --step-seconds 300     # 288 points/day

The step must divide 86400 evenly. DDRE-33 is native 15 min and Ausgrid native
30 min; going *coarser* uses block averages so daily energy is preserved, going
*finer* uses linear interpolation. Either way the sub-step ramps of the source
data cannot be recovered, so the measured ramp statistics are only meaningful at
or below the native resolution.

Note for ``run_rms.py``: its ``--time`` must land exactly on one of the generated
timestamps, otherwise the QSTS lookup fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RAW_DDRE33 = RAW / "ddre33"

# ---------------------------------------------------------------------------
# IEEE-13 feeder topology (carried locally; no dependency on the baseline case)
# ---------------------------------------------------------------------------

IEEE13_URL = "https://raw.githubusercontent.com/dss-extensions/electricdss-tst/master/Version8/Distrib/IEEETestCases"
HOURS = list(range(24))

BUS = {
    "SourceBus": [1, 2, 3], "650": [1, 2, 3], "RG60": [1, 2, 3], "632": [1, 2, 3],
    "633": [1, 2, 3], "634": [1, 2, 3], "645": [2, 3], "646": [2, 3],
    "670": [1, 2, 3], "671": [1, 2, 3], "680": [1, 2, 3], "692": [1, 2, 3],
    "675": [1, 2, 3], "684": [1, 3], "611": [3], "652": [1],
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

# Rated P/Q of each IEEE-13 load, in LOADS order.
LOAD_RATINGS = {"671": (1155, 660), "634a": (160, 110), "634b": (120, 90), "634c": (120, 90), "645": (170, 125), "646": (230, 132), "692": (170, 151), "675a": (485, 190), "675b": (68, 60), "675c": (290, 212), "611": (170, 80), "652": (128, 86), "670a": (17, 10), "670b": (66, 38), "670c": (117, 68)}

# The IEEE-13 reference 24-point daily shape. Used ONLY as the energy anchor for
# scaling the measured Ausgrid curve -- it is never emitted as a load profile.
LOAD_SHAPE = [.55, .52, .5, .5, .53, .62, .72, .82, .9, .96, 1, 1.02, 1, 1.01, 1.05, 1.1, 1.18, 1.25, 1.3, 1.28, 1.15, .95, .75, .65]
LOAD_PHASE_ADJUST = {"675a": .08, "675b": -.05, "675c": .03}

DERS = [
    {"id": "PV_632_A", "kind": "pv", "bus": "632", "phases": [1], "terminal_nodes": [1], "connection": "wye", "kv": 2.4, "p_rated_kw": 70}, {"id": "PV_670_C", "kind": "pv", "bus": "670", "phases": [3], "terminal_nodes": [3], "connection": "wye", "kv": 2.4, "p_rated_kw": 50},
    {"id": "PV_671_A", "kind": "pv", "bus": "671", "phases": [1], "terminal_nodes": [1], "connection": "wye", "kv": 2.4, "p_rated_kw": 80}, {"id": "PV_675_A", "kind": "pv", "bus": "675", "phases": [1], "terminal_nodes": [1], "connection": "wye", "kv": 2.4, "p_rated_kw": 100},
    {"id": "PV_675_C", "kind": "pv", "bus": "675", "phases": [3], "terminal_nodes": [3], "connection": "wye", "kv": 2.4, "p_rated_kw": 70}, {"id": "PV_634_ABC", "kind": "pv", "bus": "634", "phases": [1, 2, 3], "terminal_nodes": [1, 2, 3], "connection": "wye", "kv": .48, "p_rated_kw": 180},
    {"id": "BESS_671_B", "kind": "bess", "bus": "671", "phases": [2], "terminal_nodes": [2], "connection": "wye", "kv": 2.4, "p_rated_kw": 100, "energy_kwh": 400, "reserve_soc": 20, "initial_soc": 55}, {"id": "BESS_645_B", "kind": "bess", "bus": "645", "phases": [2], "terminal_nodes": [2], "connection": "wye", "kv": 2.4, "p_rated_kw": 60, "energy_kwh": 240, "reserve_soc": 20, "initial_soc": 60},
    {"id": "BESS_684_A", "kind": "bess", "bus": "684", "phases": [1], "terminal_nodes": [1], "connection": "wye", "kv": 2.4, "p_rated_kw": 50, "energy_kwh": 200, "reserve_soc": 20, "initial_soc": 50}, {"id": "EV_671_C", "kind": "ev", "bus": "671", "phases": [3], "terminal_nodes": [3], "connection": "wye", "kv": 2.4, "p_rated_kw": 60},
    {"id": "EV_675_B", "kind": "ev", "bus": "675", "phases": [2], "terminal_nodes": [2], "connection": "wye", "kv": 2.4, "p_rated_kw": 50}, {"id": "EV_611_C", "kind": "ev", "bus": "611", "phases": [3], "terminal_nodes": [3], "connection": "wye", "kv": 2.4, "p_rated_kw": 40},
    {"id": "Wind_680_ABC", "kind": "wind", "bus": "680", "phases": [1, 2, 3], "terminal_nodes": [1, 2, 3], "connection": "wye", "kv": 4.16, "p_rated_kw": 120},
]

INVERTER_S_RATIO = 1.05  # S_inv = 1.05 * P_dc_rated (typical DC/AC oversizing)

# RMS dynamic configuration. Historically this file was only ever hand-written
# and copied around, so it is inlined here as data.
RMS_CONFIG = {
    "nominal_frequency_hz": 60.0,
    "dynamic_devices": [
        {"device_id": "PV_671_A", "model_type": "gfl_inverter", "current_limit_pu": 1.2, "pll_kp_hz_per_rad": 8.0, "pll_ki_hz_per_rad_s": 80.0, "tau_p_control_s": 0.08, "tau_q_control_s": 0.08, "tau_current_s": 0.015},
        {"device_id": "BESS_671_B", "model_type": "gfm_inverter", "current_limit_pu": 1.2, "inertia_s": 2.0, "damping_pu_per_hz": 1.0, "p_droop_pu_per_hz": 0.25, "q_droop_pu_per_pu": 2.0, "voltage_droop_pu_per_pu": 0.05, "tau_power_control_s": 0.08, "tau_power_measure_s": 0.04, "tau_voltage_control_s": 0.05},
        {"device_id": "Wind_680_ABC", "model_type": "aggregate_der", "current_limit_pu": 1.15, "tau_p_control_s": 0.25, "tau_q_control_s": 0.15, "tau_measure_s": 0.05, "voltage_support_pu_per_pu": 1.5},
    ],
    "scenarios": {
        "flat_run": {"dt_s": 0.01, "t_end_s": 2.0, "events": []},
        "pv_trip": {"dt_s": 0.01, "t_end_s": 2.0, "events": [{"time_s": 1.0, "type": "der_trip", "device_id": "PV_671_A"}]},
        "load_step": {"dt_s": 0.01, "t_end_s": 2.0, "events": [{"time_s": 1.0, "type": "load_scale", "device_id": "671", "p_multiplier": 1.1, "q_multiplier": 1.1}]},
    },
}

# ---------------------------------------------------------------------------
# Simulation grid
# ---------------------------------------------------------------------------

RAW_STEP_SECONDS = 900      # native DDRE-33 resolution (verified: 35039 diffs, all 15 min);
                            # fixed by the dataset -- never derive the output step from it
RAW_STEPS_PER_DAY = 86400 // RAW_STEP_SECONDS

DEFAULT_STEP_SECONDS = 900  # simulation step, overridable with --step-seconds.


def _steps_per_day(step_seconds: int) -> int:
    """Points per day. The step must divide 86400 exactly, otherwise the last
    (partial) step of the day would silently drop part of the day."""
    if step_seconds <= 0 or 86400 % step_seconds:
        raise SystemExit(
            f"--step-seconds must divide 86400 evenly; got {step_seconds}. "
            "Use e.g. 60, 300, 600, 900, 1800 or 3600."
        )
    return 86400 // step_seconds

BESS_CHARGE_PU_THRESHOLD = 0.55
BESS_DISCHARGE_PU_THRESHOLD = 0.02
BESS_DISCHARGE_START_H = 17

SOURCE_URL = "https://github.com/YuxuanCEE/DDRE-33-CHIME"
PAPER_DOI = "10.1038/s41597-025-06464-w"

AUSGRID_FILE = RAW / "ausgrid" / "customer12_daily_pivot_cons_2011-2012.csv"
# Southern Mar-May (autumn/early winter) mirrors the northern-hemisphere Sep day of DDRE-33.
AUSGRID_MONTHS = ("2012-03", "2012-04", "2012-05")
# A feeder load aggregates many households. Scaling a *single* household curve up to a
# 1 MW load would also scale its kettle-scale spikes (measured: a 1067 kW step on a
# 1025 kW load, i.e. >100% of its own mean). Averaging N independent measured days
# keeps the real daily shape but restores spatial aggregation smoothing.
AGGREGATE_HOUSEHOLDS = 6
AUSGRID_SOURCE = "https://github.com/pierre-haessig/solarhome-control-bench"
AUSGRID_DOI = "10.1080/14786451.2015.1100196"

EV_FILE = RAW / "ev_norway" / "Dataset1_charging_reports.csv"
EV_SOURCE = "https://zenodo.org/records/13896176"
EV_DOI = "10.1016/j.dib.2024.110883"
EV_ARRIVAL_HOUR_MIN = 15.0  # keep residential evening sessions
EV_ENERGY_MIN_KWH = 3.0
EV_ENERGY_MAX_KWH = 60.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

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


def _ders() -> list[dict]:
    devices = [dict(device) for device in DERS]
    for device in devices:
        device["profile_file"] = f"der_profiles/{device['id'].lower()}.csv"
    return devices


def _network(ders: list[dict]) -> dict:
    nominal = {bus: 115.0 if bus == "SourceBus" else .48 if bus == "634" else 4.16 for bus in BUS}
    load_devices = [{"id": ident, "kind": "load", "bus": bus, "phases": nodes, "terminal_nodes": nodes, "connection": connection, "load_model": model, "kv": kv, "base_p_kw": p, "base_q_kvar": q} for (ident, bus, nodes, connection, model, kv), (p, q) in zip(LOADS, [LOAD_RATINGS[ident] for ident, *_ in LOADS], strict=True)]
    switch = {"id": "671692", "kind": "switch", "from_bus": "671", "to_bus": "692", "from_terminal_nodes": [1, 2, 3], "to_terminal_nodes": [1, 2, 3], "phases": [1, 2, 3], "normally_closed": True, "r1_ohm": 1e-4, "r0_ohm": 1e-4, "x1_ohm": 0.0, "x0_ohm": 0.0, "normal_amps": 600.0, "rating_source": "engineering_screening_assumption", "in_service": True}
    return {"schema_version": "1.0", "base": {"frequency_hz": 60, "base_kv_ll": 4.16, "slack_bus": "SourceBus"}, "constraints": {"voltage_min_pu": .95, "voltage_max_pu": 1.05, "voltage_unbalance_limit_pct": 2.0, "line_loading_limit_pct": 100.0, "transformer_loading_limit_pct": 100.0, "reverse_power_tolerance_kw": 1e-6}, "source": {"id": "Source", "bus": "SourceBus", "phases": [1, 2, 3], "base_kv_ll": 115.0, "voltage_pu": 1.0001, "angle_deg": 30.0, "mvasc3": 20000.0, "mvasc1": 21000.0}, "simulation": {"start_datetime": "2026-01-01 00:00:00", "end_datetime": "2026-01-02 00:00:00", "step_seconds": 3600}, "buses": [{"id": bus, "phases": phases, "nominal_kv_ll": nominal[bus], "is_slack": bus == "SourceBus"} for bus, phases in BUS.items()], "line_codes": LINE_CODES, "branches": [*_transformer_branches(), *[_line_branch(entry) for entry in LINE_DATA], switch], "shunt_devices": [{"id": "Cap1", "kind": "capacitor", "bus": "675", "terminal_nodes": [1, 2, 3], "phases": [1, 2, 3], "connection": "wye", "kv_ll": 4.16, "q_kvar": 600.0, "in_service": True}, {"id": "Cap2", "kind": "capacitor", "bus": "611", "terminal_nodes": [3], "phases": [3], "connection": "wye", "kv_ln": 2.4, "q_kvar": 100.0, "in_service": True}], "devices": [*load_devices, *ders]}


def _write_network(case: Path, start: pd.Timestamp, step_seconds: int) -> tuple[list[dict], list[dict]]:
    network = _network(_ders())
    end = start + timedelta(seconds=_steps_per_day(step_seconds) * step_seconds)
    network["simulation"] = {"start_datetime": start.strftime("%Y-%m-%d %H:%M:%S"), "end_datetime": end.strftime("%Y-%m-%d %H:%M:%S"), "step_seconds": step_seconds}
    devices = network["devices"]
    ders = [device for device in devices if device.get("kind") in {"pv", "wind", "bess", "ev"}]
    for device in ders:
        if device["kind"] in {"pv", "wind"}:
            device["s_rated_kva"] = round(float(device["p_rated_kw"]) * INVERTER_S_RATIO, 3)
    (case / "network.json").write_text(json.dumps(network, indent=2), encoding="utf-8")
    return devices, ders


def _fetch_ieee13_base(case: Path) -> None:
    """Download the reference IEEE-13 DSS into this case's own ieee13_base copy."""
    base = case / "ieee13_base"
    base.mkdir(parents=True, exist_ok=True)
    master, codes = base / "IEEE13Nodeckt.dss", base / "IEEELineCodes.DSS"
    if not master.exists():
        urlretrieve(IEEE13_URL + "/13Bus/IEEE13Nodeckt.dss", master)
    if not codes.exists():
        urlretrieve(IEEE13_URL + "/IEEELineCodes.DSS", codes)
    master.write_text(master.read_text(encoding="utf-8").replace("BusCoords IEEE13Node_BusXY.csv", "! Bus coordinates omitted; see network_topology.png"), encoding="utf-8")


def _write_dss(case: Path, ders: list[dict]) -> None:
    _fetch_ieee13_base(case)
    master = (case / "ieee13_base" / "IEEE13Nodeckt.dss").resolve().as_posix()
    dss = [f'Redirect "{master}"']
    for device in ders:
        phases, count, kv = ".".join(map(str, device["terminal_nodes"])), len(device["terminal_nodes"]), device["kv"]
        if device["kind"] in {"pv", "wind"}:
            dss.append(f"New Generator.{device['id']} phases={count} bus1={device['bus']}.{phases} conn={device['connection']} kV={kv} kW=0 kvar=0")
        elif device["kind"] == "ev":
            dss.append(f"New Load.{device['id']} phases={count} bus1={device['bus']}.{phases} conn=wye kV={kv} kW=0 kvar=0")
        else:
            dss.append(f"New Storage.{device['id']} phases={count} bus1={device['bus']}.{phases} conn={device['connection']} kV={kv} kWrated={device['p_rated_kw']} kWhrated={device['energy_kwh']} %stored={device['initial_soc']} %reserve={device['reserve_soc']} dispmode=EXTERNAL kW=0 kvar=0")
    (case / "network.dss").write_text("\n".join([*dss, "Solve", ""]), encoding="utf-8")


# ---------------------------------------------------------------------------
# DDRE-33 PV / wind
# ---------------------------------------------------------------------------

def _read_ddre33(name: str) -> pd.Series:
    """Return the per-unit ``OT`` column indexed by a 15-minute DatetimeIndex."""
    path = RAW_DDRE33 / name
    if not path.exists():
        raise SystemExit(f"Missing raw DDRE-33 file: {path}. Download it from {SOURCE_URL}.")
    frame = pd.read_csv(path)
    stamps = pd.to_datetime(frame["timestamp"], format="%Y/%m/%d %H:%M")
    return pd.Series(frame["OT"].to_numpy(float), index=pd.DatetimeIndex(stamps), name=path.name)


def _day_table(series: pd.Series) -> pd.DataFrame:
    """Reshape the raw series into a day x RAW_STEPS_PER_DAY matrix with day-start index."""
    values = series.to_numpy(float)
    days = len(values) // RAW_STEPS_PER_DAY
    matrix = values[: days * RAW_STEPS_PER_DAY].reshape(days, RAW_STEPS_PER_DAY)
    starts = series.index[: days * RAW_STEPS_PER_DAY : RAW_STEPS_PER_DAY]
    return pd.DataFrame(matrix, index=pd.DatetimeIndex(starts).normalize())


MIN_DAY_SPAN_H = 8.0     # reject days whose non-zero window is implausibly short (missing data)
MIN_DAY_PEAK_PU = 0.30   # reject days that never produce meaningful output
MAX_STEP_PU = 0.25       # reject days with a single-step jump above this (sensor gap, not a cloud).
                         # Defined per *raw* step, so it is independent of STEP_SECONDS.


def _resample_day(row: np.ndarray, to_steps: int) -> np.ndarray:
    """Resample one day from the raw grid to ``to_steps`` points.

    Downsampling uses block means so daily energy is preserved; upsampling uses
    linear interpolation. A no-op when the two grids already match.
    """
    if len(row) == to_steps:
        return row
    if to_steps > len(row):
        return np.interp(np.linspace(0, len(row) - 1, to_steps), np.arange(len(row)), row)
    factor = len(row) // to_steps
    return row[: to_steps * factor].reshape(to_steps, factor).mean(axis=1)


def _pick_day(pv_days: pd.DataFrame, wind_days: pd.DataFrame, choice: int | None) -> tuple[int, dict]:
    """Pick the most variable *complete* day.

    Raw DDRE-33 contains days with missing blocks (flat zeros followed by a step
    change). Those days score highest on naive ramp energy but are data gaps, not
    weather, so they are rejected before ranking.
    """
    values = pv_days.to_numpy(float)
    span_h = (values > 1e-3).sum(axis=1) * (RAW_STEP_SECONDS / 3600.0)
    peak = values.max(axis=1)
    max_step = np.abs(np.diff(values, axis=1)).max(axis=1)
    ramps = np.abs(np.diff(values, axis=1)).sum(axis=1)
    valid = (span_h >= MIN_DAY_SPAN_H) & (peak >= MIN_DAY_PEAK_PU) & (max_step <= MAX_STEP_PU)
    if choice is not None:
        index = int(choice)
        gate = {"span_h": float(span_h[index]), "peak_pu": float(peak[index]), "max_step_pu": float(max_step[index]), "passed": bool(valid[index]), "source": "user-specified"}
        return index, gate
    if not valid.any():
        raise SystemExit("No DDRE-33 day passes the quality gate; relax MIN_DAY_SPAN_H / MAX_STEP_PU.")
    index = int(np.argmax(np.where(valid, ramps, -1.0)))
    gate = {"span_h": float(span_h[index]), "peak_pu": float(peak[index]), "max_step_pu": float(max_step[index]), "passed": True, "rejected_days": int((~valid).sum()), "source": "most variable day passing the quality gate"}
    return index, gate


def _ramp_stats(p_kw: np.ndarray, dt_minutes: float) -> dict[str, float]:
    diffs = np.diff(p_kw) / dt_minutes
    return {"max_ramp_kw_per_min": float(np.max(np.abs(diffs))), "mean_abs_ramp_kw_per_min": float(np.mean(np.abs(diffs))), "std_kw": float(np.std(p_kw)), "peak_kw": float(np.max(p_kw))}


def _bess_dispatch(fleet_pu: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Rule-based schedule: absorb midday PV surplus, discharge into the evening."""
    command = np.zeros(len(fleet_pu))
    charge = fleet_pu > BESS_CHARGE_PU_THRESHOLD
    discharge = (fleet_pu < BESS_DISCHARGE_PU_THRESHOLD) & (hours >= BESS_DISCHARGE_START_H)
    command[charge] = -0.55
    command[discharge] = 0.60
    return command


# ---------------------------------------------------------------------------
# Measured load (Ausgrid)
# ---------------------------------------------------------------------------

def _reference_load_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """24-point P/Q reference for each load, used only as the energy anchor."""
    curves = {}
    for ident, *_ in LOADS:
        p_kw, q_kvar = LOAD_RATINGS[ident]
        adjust = 1.0 + LOAD_PHASE_ADJUST.get(ident, 0)
        curves[ident] = (
            np.array([round(p_kw * LOAD_SHAPE[hour] * adjust, 3) for hour in HOURS]),
            np.array([round(q_kvar * LOAD_SHAPE[hour] * adjust, 3) for hour in HOURS]),
        )
    return curves


def _ausgrid_days() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (hour grid, [n_days x 48] kW matrix, dates) for the chosen season."""
    frame = pd.read_csv(AUSGRID_FILE)
    hours = np.array([float(column) for column in frame.columns[1:]], dtype=float)
    dates = frame["date"].astype(str)
    season = frame[dates.str.startswith(AUSGRID_MONTHS)]
    if season.empty:
        raise SystemExit(f"No Ausgrid rows for {AUSGRID_MONTHS}; adjust AUSGRID_MONTHS.")
    return hours, season.iloc[:, 1:].to_numpy(float), season["date"].astype(str).tolist()


def _load_profiles_measured(timestamps: pd.DatetimeIndex, steps_per_day: int) -> tuple[pd.DataFrame, list[str]]:
    """Replace the synthetic 24-point load shape with measured Ausgrid curves."""
    hours, curves, dates = _ausgrid_days()
    grid = np.arange(steps_per_day) * (24.0 / steps_per_day)
    # wrap the daily curve so interpolation is valid right up to 24:00
    x = np.append(hours, 24.0)

    reference = _reference_load_curves()
    rows, used_dates = [], []
    for position, (name, *_) in enumerate(LOADS):
        base_p, base_q = reference[name]
        # match the reference's own interpolation so daily energy is preserved exactly
        base_p_curve = np.interp(grid, np.arange(25.0), np.append(base_p, base_p[0]))
        base_q_curve = np.interp(grid, np.arange(25.0), np.append(base_q, base_q[0]))
        base_p_mean = float(base_p_curve.mean())
        base_q_mean = float(base_q_curve.mean())

        # aggregate N independent measured days -> real daily shape, smoothed like a real feeder load
        start = (position * AGGREGATE_HOUSEHOLDS) % len(curves)
        picks = [(start + k) % len(curves) for k in range(AGGREGATE_HOUSEHOLDS)]
        daily = curves[picks].mean(axis=0)
        curve = np.append(daily, daily[0])
        p_shape = np.interp(grid, x, curve)
        scale = base_p_mean / float(p_shape.mean())  # preserve daily energy
        q_ratio = base_q_mean / base_p_mean if base_p_mean else 0.0

        rows.append(pd.DataFrame({"timestamp": timestamps, "load_name": name, "p_kw": (p_shape * scale).round(3), "q_kvar": (p_shape * scale * q_ratio).round(3)}))
        used_dates.append(f"{dates[picks[0]]}..{dates[picks[-1]]} ({AGGREGATE_HOUSEHOLDS} days)")
    return pd.concat(rows, ignore_index=True), used_dates


# ---------------------------------------------------------------------------
# Measured EV (Norway)
# ---------------------------------------------------------------------------

def _ev_sessions(rng: np.random.Generator, count: int) -> pd.DataFrame:
    """Draw real residential charging sessions (Norway, 2019)."""
    frame = pd.read_csv(EV_FILE, sep=";", decimal=",")
    frame["plugin"] = pd.to_datetime(frame["plugin_time"])
    frame["energy_kwh"] = frame["energy_session"].astype(float)
    frame["arrival_h"] = frame["plugin"].dt.hour + frame["plugin"].dt.minute / 60.0
    keep = (frame["arrival_h"] >= EV_ARRIVAL_HOUR_MIN) & frame["energy_kwh"].between(EV_ENERGY_MIN_KWH, EV_ENERGY_MAX_KWH)
    pool = frame[keep]
    if len(pool) < count:
        raise SystemExit(f"EV session pool too small: {len(pool)} < {count}")
    picked = pool.iloc[rng.choice(len(pool), size=count, replace=False)]
    return picked.reset_index(drop=True)


def _ev_power(session: pd.Series, rating_kw: float, hours: np.ndarray, dt_hours: float) -> np.ndarray:
    """Constant-power charge from the measured plug-in time until the measured energy is delivered."""
    power = np.zeros(len(hours))
    arrival = float(session["arrival_h"])
    demand_kwh = float(session["energy_kwh"])
    energy = 0.0
    for index in range(len(hours)):
        if hours[index] + dt_hours <= arrival:
            continue
        if energy >= demand_kwh:
            break
        step_energy = min(rating_kw * dt_hours, demand_kwh - energy)
        power[index] = step_energy / dt_hours
        energy += step_energy
    return power


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate an IEEE-13 case driven entirely by measured time series.")
    parser.add_argument("--day", type=int, default=None, help="DDRE-33 day index (default: most variable day).")
    parser.add_argument("--seed", type=int, default=20250701, help="RNG seed for EV session pick.")
    parser.add_argument("--name", default="ieee13_measured", help="Output case directory name.")
    parser.add_argument("--step-seconds", type=int, default=DEFAULT_STEP_SECONDS,
                        help="Simulation step in seconds (default 900). Must divide 86400: e.g. 60, 300, 900, 1800, 3600. "
                             "Coarser steps are produced by energy-preserving block averages, so sub-step ramps are lost.")
    args = parser.parse_args(argv)

    step_seconds = args.step_seconds
    steps_per_day = _steps_per_day(step_seconds)

    pv = _read_ddre33("solar33_real_1.csv")
    wind = _read_ddre33("wind22_real_1.csv")
    pv_days, wind_days = _day_table(pv), _day_table(wind)
    day, gate = _pick_day(pv_days, wind_days, args.day)
    day_start = pv_days.index[day]
    pv_pu = _resample_day(pv_days.iloc[day].to_numpy(float), steps_per_day)
    wind_pu = _resample_day(wind_days.iloc[day].to_numpy(float), steps_per_day)

    timestamps = pd.DatetimeIndex([day_start + timedelta(seconds=step_seconds * step) for step in range(steps_per_day)])
    stamp = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
    dt_hours = step_seconds / 3600.0
    dt_minutes = step_seconds / 60.0

    case = ROOT / "data" / args.name
    case.mkdir(parents=True, exist_ok=True)
    profiles = case / "der_profiles"
    profiles.mkdir(exist_ok=True)

    devices, ders = _write_network(case, timestamps[0], step_seconds)
    _write_dss(case, ders)
    # write_bytes (not write_text) so the file keeps LF endings on Windows, matching
    # the hand-authored rms_config.json used by the baseline case.
    (case / "rms_config.json").write_bytes((json.dumps(RMS_CONFIG, indent=2) + "\n").encode("utf-8"))

    load_frame, load_dates = _load_profiles_measured(timestamps, steps_per_day)
    load_frame.to_csv(case / "load_profiles.csv", index=False)

    rng = np.random.default_rng(args.seed)
    ev_count = sum(1 for d in ders if d["kind"] == "ev")
    ev_pool = _ev_sessions(rng, ev_count)
    ev_used: dict[str, dict] = {}
    cursor = 0

    fleet_capacity = sum(float(d["p_rated_kw"]) for d in ders if d["kind"] == "pv")
    summary: dict[str, dict[str, float]] = {}

    for device in ders:
        kind, ident, rating = device["kind"], device["id"], float(device["p_rated_kw"])
        if kind == "pv":
            p_kw = pv_pu * rating
            s_kva = rating * INVERTER_S_RATIO
            q_limit = np.sqrt(np.clip(s_kva**2 - p_kw**2, 0.0, None))
            frame = pd.DataFrame({"timestamp": stamp, "pv_kw": p_kw.round(4), "pv_kvar": 0.0, "controller_type": "volt_var", "q_limit_kvar": q_limit.round(4), "v_ref_pu": 1.0, "droop_kvar_per_pu": rating * 5})
        elif kind == "wind":
            p_kw = wind_pu * rating
            frame = pd.DataFrame({"timestamp": stamp, "wind_kw": p_kw.round(4), "wind_kvar": 0.0, "controller_type": "constant_pq"})
        elif kind == "bess":
            p_kw = _bess_dispatch(pv_pu, hours) * rating
            frame = pd.DataFrame({"timestamp": stamp, "bess_kw": p_kw.round(4), "bess_kvar": 0.0, "controller_type": "soc_schedule"})
        else:
            session = ev_pool.iloc[cursor]
            cursor += 1
            p_kw = _ev_power(session, rating, hours, dt_hours)
            frame = pd.DataFrame({"timestamp": stamp, "ev_kw": p_kw.round(4), "ev_kvar": 0.0, "controller_type": "constant_pq"})
            ev_used[ident] = {"plugin_local": str(session["plugin_time"]), "energy_kwh": round(float(session["energy_kwh"]), 2), "connection_h": round(float(session["connection_time"]), 2), "user_id": str(session["user_id"]), "site": str(session["location"])}
        frame.to_csv(profiles / f"{ident.lower()}.csv", index=False)
        summary[ident] = {"kind": kind, "p_rated_kw": rating, **_ramp_stats(p_kw, dt_minutes)}

    fleet_pv = sum(summary[d["id"]]["peak_kw"] for d in ders if d["kind"] == "pv")
    meta = {
        "case": args.name,
        "timeline": {"start": stamp[0], "end": stamp[-1], "step_seconds": step_seconds, "points_per_day": steps_per_day},
        "day_quality_gate": gate,
        "sources": {
            "pv": {"dataset": "DDRE-33 node_33 PV", "file": "data/raw/ddre33/solar33_real_1.csv", "sha256": _sha256(RAW_DDRE33 / "solar33_real_1.csv"), "doi": PAPER_DOI},
            "wind": {"dataset": "DDRE-33 node_22 wind", "file": "data/raw/ddre33/wind22_real_1.csv", "sha256": _sha256(RAW_DDRE33 / "wind22_real_1.csv"), "doi": PAPER_DOI},
            "load": {"dataset": f"Ausgrid solar home electricity, customer 12 (30 min), {AGGREGATE_HOUSEHOLDS}-day aggregate", "file": "data/raw/ausgrid/customer12_daily_pivot_cons_2011-2012.csv", "sha256": _sha256(AUSGRID_FILE), "url": AUSGRID_SOURCE, "doi": AUSGRID_DOI, "months_used": list(AUSGRID_MONTHS), "aggregate_households": AGGREGATE_HOUSEHOLDS, "dates_used": load_dates},
            "ev": {"dataset": "Residential EV charging, 12 sites in Norway (35k sessions)", "file": "data/raw/ev_norway/Dataset1_charging_reports.csv", "sha256": _sha256(EV_FILE), "url": EV_SOURCE, "doi": EV_DOI, "sessions": ev_used},
            "bess": {"dataset": "rule-based dispatch on the measured PV trace", "file": None, "doi": None},
        },
        "derivation": {
            "pv_kw": "pu trace x p_rated_kw",
            "wind_kw": "pu trace x p_rated_kw",
            "load_kw": "Ausgrid 30-min curve resampled to the case grid, rescaled so daily mean power equals the IEEE-13 load's daily mean (energy preserving); Q uses that load's own Q/P ratio",
            "ev_kw": "constant power from the measured plug-in time until the measured session energy is delivered, capped at p_rated_kw",
            "q_limit_kvar": "sqrt(S^2 - P^2), S = 1.05 x p_rated_kw",
        },
        "limitations": [
            "Sources are not co-located and not observed on the same day: this is a representative day assembled from independent measurements.",
            "Ausgrid contributes a single customer, so the 15 feeder loads reuse it on 15 different dates.",
            "Ausgrid dates come from 2012-03 (southern autumn) to match the northern-autumn DDRE-33 day.",
            "BESS dispatch is still a rule-based schedule, not a measured trace.",
            "EV charging is constant power; real sessions taper near full SOC (CC-CV), which is not modelled.",
        ],
    }
    (case / "profile_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"case              : {case}")
    print(f"day               : index {day} -> {stamp[0]} .. {stamp[-1]} ({steps_per_day} x {step_seconds} s)")
    print(f"PV peak (fleet)   : {fleet_pv:.1f} kW of {fleet_capacity:.0f} kW installed")
    print(f"load source       : Ausgrid customer 12, {'/'.join(AUSGRID_MONTHS)} ({len(load_dates)} loads x {AGGREGATE_HOUSEHOLDS} real days)")
    print("EV sessions       :")
    for ident, info in ev_used.items():
        print(f"  {ident:<14s} plugin={info['plugin_local']}  energy={info['energy_kwh']:5.2f} kWh  connected={info['connection_h']:5.2f} h  site={info['site']}")
    print("per-device        :")
    for ident, stats in summary.items():
        print(f"  {ident:<14s} peak={stats['peak_kw']:7.2f} kW  max|ramp|={stats['max_ramp_kw_per_min']:6.2f} kW/min  std={stats['std_kw']:6.2f} kW")


if __name__ == "__main__":
    _main()
