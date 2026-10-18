"""Build an IEEE-13 case whose DER output comes from ERA5 weather via the pvlib chain.

This is stage 1 of the ``ieee13_era5`` plan: a minimal closed loop that proves the
whole chain works before the stochastic layer (ARIMA + Copula) is added.

    ERA5 hourly reanalysis (Open-Meteo)
        -> pvlib ModelChain          -> PV AC power
        -> 10 m wind -> hub height   -> wind power curve
        -> IEEE-13 QSTS

Randomness at this stage comes from the weather itself: every day of the requested
year is a real observed day, so no synthetic generation is applied yet. The
ARIMA/Copula layer is deliberately absent until this loop is validated.

Modes
-----
    # single runnable case (then use run.py / run_rms.py as usual)
    python data/generate_data/ieee13_era5.py --date 2022-09-20

    # batch: run N consecutive days in-process and emit a statistics table
    python data/generate_data/ieee13_era5.py --scenarios 100 --start 2022-01-01

    # same, but driven by the ARIMA(kt)+copula generator instead of replayed days
    python data/generate_data/ieee13_era5.py --weather synth --scenarios 100 --start 2022-01-01

Weather source
--------------
``--weather era5`` replays observed days: one run = one real day, and the only
randomness is which historical day was drawn. ``--weather synth`` draws from the
generative model in ``era5_synth.py`` (fitted on 1990-2014, validated on 2015-2024),
so the scenario set can be larger than the record. pvlib stays the deterministic
weather -> power converter in both cases.

Assumptions (also written to profile_meta.json)
-----------------------------------------------
* Site coordinates: IEEE-13 is a standard test feeder with no location, so a site
  is chosen explicitly and documented. ``--site`` selects among three US climates.
* surface_tilt = site latitude (physical optimum for fixed mount, annual energy).
* surface_azimuth = 180 deg (due south) +- AZIMUTH_SPREAD_DEG, sampled once per
  case from a seeded RNG, representing dispersed residential roof orientations.
* Temperature model ``close_mount_glass_glass``: roof-mounted, poor ventilation
  (``open_rack_glass_glass`` would be wrong -- that is ground-mounted).
* DC/AC ratio 1.15, implemented per pvlib semantics as
  ``inverter pdc0 = module_pdc0 / (ratio * inverter_eta_nom)``. Note that this
  ratio clips the peak but leaves daily energy essentially unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from urllib.request import urlretrieve  # noqa: E402

import pvlib  # noqa: E402
from pvlib.location import Location  # noqa: E402
from pvlib.modelchain import ModelChain  # noqa: E402
from pvlib.pvsystem import PVSystem  # noqa: E402

from src.power_flow import NetworkModel  # noqa: E402
from src.profile_store import ProfileStore  # noqa: E402
from src.qsts import run_qsts  # noqa: E402
from src.solvers.opendss_solver import OpenDSSSolver  # noqa: E402


RAW = ROOT / "data" / "raw"
ERA5_CACHE = RAW / "era5"


# -------------------------------------------------------------------------
# IEEE-13 shared plumbing (topology / network writer / load / EV / BESS),
# inlined verbatim from ieee13_measured.py so this generator is self-contained
# and does NOT import the measured generator.
# -------------------------------------------------------------------------

IEEE13_URL = "https://raw.githubusercontent.com/dss-extensions/electricdss-tst/master/Version8/Distrib/IEEETestCases"

HOURS = list(range(24))

BUS = {
    "SourceBus": [1, 2, 3], "650": [1, 2, 3], "RG60": [1, 2, 3], "632": [1, 2, 3],
    "633": [1, 2, 3], "634": [1, 2, 3], "645": [2, 3], "646": [2, 3],
    "670": [1, 2, 3], "671": [1, 2, 3], "680": [1, 2, 3], "692": [1, 2, 3],
    "675": [1, 2, 3], "684": [1, 3], "611": [3], "652": [1],
}

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

LOAD_RATINGS = {"671": (1155, 660), "634a": (160, 110), "634b": (120, 90), "634c": (120, 90), "645": (170, 125), "646": (230, 132), "692": (170, 151), "675a": (485, 190), "675b": (68, 60), "675c": (290, 212), "611": (170, 80), "652": (128, 86), "670a": (17, 10), "670b": (66, 38), "670c": (117, 68)}

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

AUSGRID_FILE = RAW / "ausgrid" / "customer12_daily_pivot_cons_2011-2012.csv"

AUSGRID_MONTHS = ("2012-03", "2012-04", "2012-05")

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
    for target in (master, codes):
        if target.exists():
            continue
        try:
            urlretrieve(IEEE13_URL + "/" + target.name, target)
        except Exception:
            # The upstream repo drops connections often enough to make a fresh case
            # fail outright. Any case already on disk carries an identical copy of
            # these two static files, so reuse one instead of re-downloading.
            local = sorted((ROOT / "data").glob("*/ieee13_base/" + target.name))
            if not local:
                raise
            target.write_bytes(local[0].read_bytes())
            print(f"note              : {target.name} taken from {local[0].parent.parent.name} "
                  f"(upstream download failed)")
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

def _ausgrid_days(months: tuple[str, ...] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (hour grid, [n_days x 48] kW matrix, dates) for the chosen season.

    ``months`` defaults to ``AUSGRID_MONTHS``. Multi-scenario callers (see
    ``ieee13_era5.py``) pass a single month prefix so that each scenario draws
    its load from the season matching that scenario's date.
    """
    wanted = months if months is not None else AUSGRID_MONTHS
    frame = pd.read_csv(AUSGRID_FILE)
    hours = np.array([float(column) for column in frame.columns[1:]], dtype=float)
    dates = frame["date"].astype(str)
    season = frame[dates.str.startswith(wanted)]
    if season.empty:
        raise SystemExit(f"No Ausgrid rows for {wanted}; adjust the month filter.")
    return hours, season.iloc[:, 1:].to_numpy(float), season["date"].astype(str).tolist()

def _load_profiles_measured(timestamps: pd.DatetimeIndex, steps_per_day: int,
                            months: tuple[str, ...] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Replace the synthetic 24-point load shape with measured Ausgrid curves."""
    hours, curves, dates = _ausgrid_days(months)
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

WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ("shortwave_radiation,direct_normal_irradiance,diffuse_radiation,"
          "temperature_2m,wind_speed_10m,wind_speed_100m")

# ``tz`` is the civil timezone, used only for labelling. ``solar_tz`` is what every
# solar-position calculation must use, and it is NOT the civil zone:
# Open-Meteo's archive endpoint applies ONE fixed UTC offset to the whole requested
# range (verified: utc_offset_seconds = -14400 for America/New_York in both January
# and July, and for every year back to 1990; -25200 for America/Phoenix). The
# timestamps are therefore a fixed-offset clock with no DST, not civil local time.
# Consequences of getting this wrong:
#   * ``tz_localize("America/New_York")`` raises NonExistentTimeError on the
#     spring-forward hour (the cached series still contains 02:00), so any sweep
#     covering a DST transition day used to crash;
#   * outside DST it silently puts the sun 1 h early, which shifts the whole PV curve.
# Cross-check: with ``solar_tz`` the ERA5 GHI lags pvlib's ineichen clearsky by 0-1 h
# (corr 0.91-0.97), which is exactly the "average of the preceding hour" convention
# Open-Meteo documents -- i.e. the geometry now lines up.
SITES = {
    "atlanta": {"lat": 33.65, "lon": -84.42, "tz": "America/New_York", "solar_tz": "Etc/GMT+4",
                "label": "Atlanta, GA (subtropical humid)"},
    "newyork": {"lat": 40.71, "lon": -74.01, "tz": "America/New_York", "solar_tz": "Etc/GMT+4",
                "label": "New York, NY (temperate continental)"},
    "phoenix": {"lat": 33.45, "lon": -112.07, "tz": "America/Phoenix", "solar_tz": "Etc/GMT+7",
                "label": "Phoenix, AZ (subtropical arid)"},
}

# Simulation grid: ERA5 is hourly, so the case runs at 1 h / 24 points per day.
# This also matches the baseline case, making a same-resolution comparison valid.
STEP_SECONDS = 3600
STEPS_PER_DAY = 24

PV_DC_AC_RATIO = 1.15
MODULE_GAMMA_PDC = -0.004
TEMP_MODEL = "close_mount_glass_glass"
INVERTER_ETA_NOM = 0.96
AZIMUTH_SPREAD_DEG = 30.0
TILT_JITTER_DEG = 5.0
ARRAY_SEED = 20251011

# Wind: hub height 100 m, so ERA5's native 100 m wind speed is used directly and
# no vertical extrapolation is applied (extrapolating 10 m up with a 1/7 power law
# is a second-best that ERA5 makes unnecessary).
# The power curve is a piecewise-linear typical utility-scale curve rather than a
# cubic law: a pure cubic underestimates output badly at the 5-8 m/s speeds that
# dominate inland ERA5 grid cells.
# Volt-VAR settings, IEEE 1547-2018 default curve:
#   V1=0.90 -> full injection (+Qmax)   V2=0.98 -> 0
#   V3=1.02 -> 0                        V4=1.10 -> full absorption (-Qmax)
# The deadband (V2..V3) is what makes the steep slope usable: without it the
# coupled control loop diverges once penetration rises (measured).
# The slope is defined against the *available* headroom q_limit(t), so the curve
# keeps its shape as the inverter's spare capacity changes through the day.
VOLT_VAR_V1_PU = 0.90
VOLT_VAR_V2_PU = 0.98
VOLT_VAR_V3_PU = 1.02
VOLT_VAR_V4_PU = 1.10
VOLT_VAR_DEADBAND_PU = (VOLT_VAR_V3_PU - VOLT_VAR_V2_PU) / 2.0   # 0.02
VOLT_VAR_DROOP_SPAN_PU = VOLT_VAR_V2_PU - VOLT_VAR_V1_PU         # 0.08
# Slope per unit of rated real power. Qmax ~ 1.05 * rating (inverter can source a
# little more vars than watts), so the curve reaches full injection/absorption over
# the 0.08 pu action band: droop = 1.05 / 0.08 ~ 13.1 * rating per pu. The per-step
# q_limit clip then caps it at the inverter's available reactive headroom as P varies.
VOLT_VAR_DROOP_PER_RATING = 1.05 / VOLT_VAR_DROOP_SPAN_PU         # 13.125
WIND_CURVE_MS = (0.0, 3.0, 5.0, 7.0, 9.0, 11.0, 12.0, 25.0, 30.0)
WIND_CURVE_PU = (0.0, 0.0, 0.08, 0.25, 0.50, 0.82, 1.00, 1.00, 0.0)

DEFAULT_CASE = "ieee13_era5"
EV_SEED = 20250701
# Seed for --weather synth. Distinct from EV_SEED/ARRAY_SEED so that changing one does
# not silently reshuffle the others.
SYNTH_SEED = 20251014

SOURCE_URL = "https://open-meteo.com/en/docs/historical-weather-api"
ERA5_DOI = "10.24381/cds.adbb2d47"


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def _fetch_era5(site: str, year: int) -> pd.DataFrame:
    """Download (or read from cache) one year of ERA5 hourly weather."""
    ERA5_CACHE.mkdir(parents=True, exist_ok=True)
    cache = ERA5_CACHE / f"{site}_{year}.csv"
    if cache.exists():
        frame = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(frame):
            return frame

    info = SITES[site]
    url = (
        f"{WEATHER_API}?latitude={info['lat']}&longitude={info['lon']}"
        f"&start_date={year}-01-01&end_date={year}-12-31"
        f"&hourly={HOURLY}&timezone={urllib.parse.quote(info['tz'])}&wind_speed_unit=ms"
    )
    with urllib.request.urlopen(url, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
    hourly = payload["hourly"]
    frame = pd.DataFrame(
        {
            "ghi": hourly["shortwave_radiation"],
            "dni": hourly["direct_normal_irradiance"],
            "dhi": hourly["diffuse_radiation"],
            "temp_air": hourly["temperature_2m"],
            "wind_speed": hourly["wind_speed_10m"],
            "wind_speed_100m": hourly["wind_speed_100m"],
        },
        index=pd.DatetimeIndex(pd.to_datetime(hourly["time"])),
    ).astype(float)
    if frame.isna().any().any():
        raise SystemExit(f"ERA5 response contains missing values for {site} {year}.")
    frame.to_csv(cache)
    return frame


# ---------------------------------------------------------------------------
# pvlib and wind
# ---------------------------------------------------------------------------

def _pv_pu(weather: pd.DataFrame, lat: float, lon: float, tz: str,
           tilt: float, azimuth: float) -> np.ndarray:
    """AC power per unit of installed DC capacity, via the pvlib PVWatts chain.

    ``tz`` must be the site's ``solar_tz`` (fixed UTC offset), never the civil zone:
    see the note at ``SITES``. The cached ERA5 index is wall clock on that fixed
    clock and tz-naive; pvlib reads a naive index as UTC, which used to shift the
    solar position by the whole offset (apparent elevation -16 deg at 09:00 and 78
    deg at 18:00). Localizing first fixed that, and using the fixed offset (rather
    than the DST-aware civil zone) fixes the residual 1 h winter error and the
    NonExistentTimeError on DST transition days.
    """
    if weather.index.tz is None:
        weather = weather.tz_localize(tz)
    location = Location(lat, lon, tz=tz)
    system = PVSystem(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        module_parameters={"pdc0": 1.0, "gamma_pdc": MODULE_GAMMA_PDC},
        inverter_parameters={"pdc0": 1.0 / (PV_DC_AC_RATIO * INVERTER_ETA_NOM)},
        temperature_model_parameters=pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][TEMP_MODEL],
    )
    chain = ModelChain.with_pvwatts(system, location, aoi_model="physical", spectral_model="no_loss")
    chain.run_model(weather)
    return np.clip(chain.results.ac.to_numpy(float), 0.0, None)


def _wind_pu(weather: pd.DataFrame) -> np.ndarray:
    """Wind power per unit of rated capacity, from ERA5's native 100 m wind speed."""
    v_hub = weather["wind_speed_100m"].to_numpy(float)
    return np.clip(np.interp(v_hub, WIND_CURVE_MS, WIND_CURVE_PU), 0.0, 1.0)


def _array_geometry(lat: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """tilt = latitude (+- jitter), azimuth = due south (+- spread), seeded."""
    rng = np.random.default_rng(ARRAY_SEED)
    tilts = np.clip(lat + rng.uniform(-TILT_JITTER_DEG, TILT_JITTER_DEG, count), 0.0, 90.0)
    azimuths = 180.0 + rng.uniform(-AZIMUTH_SPREAD_DEG, AZIMUTH_SPREAD_DEG, count)
    return tilts.round(1), azimuths.round(1)


# ---------------------------------------------------------------------------
# Season matching for the load
# ---------------------------------------------------------------------------

def _ausgrid_month_prefix(date: pd.Timestamp) -> str:
    """Map a Northern-Hemisphere ERA5 month onto the Ausgrid (Southern) month.

    Ausgrid covers 2011-07 .. 2012-06, so months 7-12 are 2011 and 1-6 are 2012.
    +6 months turns northern September into southern March, which is exactly the
    season matching used by ``ieee13_measured``.
    """
    shifted = (date.month + 5) % 12 + 1
    year = 2011 if shifted >= 7 else 2012
    return f"{year}-{shifted:02d}"


# ---------------------------------------------------------------------------
# Case writing
# ---------------------------------------------------------------------------

def _ensure_case(case: Path, start: pd.Timestamp) -> None:
    """Write only the case scaffolding (network.json / network.dss / rms_config.json)."""
    case.mkdir(parents=True, exist_ok=True)
    (case / "der_profiles").mkdir(exist_ok=True)
    _write_network(case, start, STEP_SECONDS)
    _write_dss(case, [d for d in _network(_ders())["devices"] if d.get("kind") in {"pv", "wind", "bess", "ev"}])
    (case / "rms_config.json").write_bytes((json.dumps(RMS_CONFIG, indent=2) + "\n").encode("utf-8"))


def _write_case(case: Path, site: str, timestamps: pd.DatetimeIndex,
                pv_pu: np.ndarray, wind_pu: np.ndarray, seed: int,
                initial_soc: float | None = None, pv_scale: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Write network/dss/rms_config/load/EV/DER files for one day.

    ``initial_soc`` overrides every BESS starting SOC, which is how the batch mode
    probes sensitivity to the initial state (single-day scenarios cannot carry SOC
    across days, so this is the only way to span plausible starting conditions).
    """
    case.mkdir(parents=True, exist_ok=True)
    (case / "der_profiles").mkdir(exist_ok=True)

    devices, ders = _write_network(case, timestamps[0], STEP_SECONDS)
    if initial_soc is not None or pv_scale != 1.0:
        path = case / "network.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for device in payload["devices"]:
            if device.get("kind") == "bess" and initial_soc is not None:
                device["initial_soc"] = initial_soc
            if device.get("kind") == "pv" and pv_scale != 1.0:
                device["p_rated_kw"] = round(float(device["p_rated_kw"]) * pv_scale, 3)
                device["s_rated_kva"] = round(float(device["p_rated_kw"]) * INVERTER_S_RATIO, 3)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ders = [d for d in payload["devices"] if d.get("kind") in {"pv", "wind", "bess", "ev"}]
    _write_dss(case, ders)
    (case / "rms_config.json").write_bytes((json.dumps(RMS_CONFIG, indent=2) + "\n").encode("utf-8"))

    stamp = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
    dt_hours = STEP_SECONDS / 3600.0
    dt_minutes = STEP_SECONDS / 60.0

    months = (_ausgrid_month_prefix(timestamps[0]),)
    load_frame, load_dates = _load_profiles_measured(timestamps, STEPS_PER_DAY, months)
    load_frame.to_csv(case / "load_profiles.csv", index=False)

    rng = np.random.default_rng(seed)
    ev_count = sum(1 for d in ders if d["kind"] == "ev")
    ev_pool = _ev_sessions(rng, ev_count)
    ev_used: dict[str, dict] = {}
    cursor = 0
    summary: dict[str, dict] = {}

    for device in ders:
        kind, ident, rating = device["kind"], device["id"], float(device["p_rated_kw"])
        if kind == "pv":
            index = [i for i, d in enumerate(ders) if d["id"] == ident][0]
            p_kw = pv_pu[index] * rating
            s_kva = rating * INVERTER_S_RATIO
            q_limit = np.sqrt(np.clip(s_kva ** 2 - p_kw ** 2, 0.0, None))
            frame = pd.DataFrame({"timestamp": stamp, "pv_kw": p_kw.round(4), "pv_kvar": 0.0,
                                  "controller_type": "volt_var", "q_limit_kvar": q_limit.round(4),
                                  "v_ref_pu": 1.0,
                                  "droop_kvar_per_pu": rating * VOLT_VAR_DROOP_PER_RATING,
                                  "deadband_pu": VOLT_VAR_DEADBAND_PU})
        elif kind == "wind":
            p_kw = wind_pu * rating
            frame = pd.DataFrame({"timestamp": stamp, "wind_kw": p_kw.round(4), "wind_kvar": 0.0,
                                  "controller_type": "constant_pq"})
        elif kind == "bess":
            p_kw = _bess_dispatch(pv_pu.mean(axis=0), hours) * rating
            frame = pd.DataFrame({"timestamp": stamp, "bess_kw": p_kw.round(4), "bess_kvar": 0.0,
                                  "controller_type": "soc_schedule"})
        else:
            session = ev_pool.iloc[cursor]
            cursor += 1
            p_kw = _ev_power(session, rating, hours, dt_hours)
            frame = pd.DataFrame({"timestamp": stamp, "ev_kw": p_kw.round(4), "ev_kvar": 0.0,
                                  "controller_type": "constant_pq"})
            ev_used[ident] = {"plugin_local": str(session["plugin_time"]),
                              "energy_kwh": round(float(session["energy_kwh"]), 2),
                              "connection_h": round(float(session["connection_time"]), 2),
                              "user_id": str(session["user_id"]), "site": str(session["location"])}
        frame.to_csv(case / "der_profiles" / f"{ident.lower()}.csv", index=False)
        summary[ident] = {"kind": kind, "p_rated_kw": rating, **_ramp_stats(p_kw, dt_minutes)}

    return devices, {"ev_used": ev_used, "load_dates": load_dates, "per_device": summary}


def _write_meta(case: Path, site: str, timestamps: pd.DatetimeIndex,
                tilts: np.ndarray, azimuths: np.ndarray, extras: dict,
                weather_mode: str = "era5") -> None:
    info = SITES[site]
    pv_rated = [float(d["p_rated_kw"]) for d in DERS if d["kind"] == "pv"]
    meta = {
        "case": case.name,
        "timeline": {"start": timestamps[0].strftime("%Y-%m-%d %H:%M:%S"),
                     "end": timestamps[-1].strftime("%Y-%m-%d %H:%M:%S"),
                     "step_seconds": STEP_SECONDS, "points_per_day": STEPS_PER_DAY},
        "sources": {
            "weather": {"dataset": f"ERA5 hourly reanalysis via Open-Meteo archive API ({info['label']})",
                        "generator": ("observed replay" if weather_mode == "era5" else
                                      "ARIMA(kt) + Gaussian copula synthetic, see era5_synth.py"),
                        "file": str(ERA5_CACHE.relative_to(ROOT)), "url": SOURCE_URL, "doi": ERA5_DOI,
                        "resolution": "1 h", "grid": "0.25 deg (~25 km)",
                        "timezone": f"fixed UTC offset {info['solar_tz']} (Open-Meteo applies no DST)"},
            "load": {"dataset": "Ausgrid solar home electricity, customer 12 (30 min)",
                     "file": "data/raw/ausgrid/customer12_daily_pivot_cons_2011-2012.csv",
                     "sha256": _sha256(AUSGRID_FILE), "doi": AUSGRID_DOI,
                     "months_used": [extras["load_dates"] and _ausgrid_month_prefix(timestamps[0])]},
            "ev": {"dataset": "Residential EV charging, 12 sites in Norway (35k sessions)",
                   "file": "data/raw/ev_norway/Dataset1_charging_reports.csv",
                   "sha256": _sha256(EV_FILE), "doi": EV_DOI, "sessions": extras["ev_used"]},
            "bess": {"dataset": "rule-based dispatch on the computed PV curve", "file": None, "doi": None},
        },
        "derivation": {
            "pv_kw": "ERA5 GHI/DNI/DHI + temperature -> pvlib ModelChain.with_pvwatts AC output, per unit of DC rating",
            "wind_kw": "ERA5 10 m wind -> 1/7 power law to 80 m -> cubic ramp between cut-in and rated",
            "load_kw": "Ausgrid 30-min curve resampled to the case grid, season-matched to the scenario month, rescaled to preserve daily energy",
            "q_limit_kvar": f"sqrt(S^2 - P^2), S = {INVERTER_S_RATIO} x p_rated_kw",
        },
        "assumptions": {
            "site_coordinates": {"value": {"latitude": info["lat"], "longitude": info["lon"], "timezone": info["tz"],
                                 "solar_timezone": info["solar_tz"]},
                                 "is_assumption": True,
                                 "reason": "IEEE-13 is a standard test feeder with no official location; a US site is assumed. Mitigated by running three climates."},
            "surface_tilt_deg": {"value": tilts.tolist(), "is_assumption": "partly",
                                 "reason": "latitude is the annual-energy optimum for fixed mount; +-5 deg jitter represents roof variation"},
            "surface_azimuth_deg": {"value": azimuths.tolist(), "is_assumption": True,
                                    "reason": "due south is optimal in the northern hemisphere; +-30 deg spread represents dispersed roof orientations and produces output diversity without a cloud model"},
            "temperature_model": {"value": TEMP_MODEL, "is_assumption": False,
                                  "reason": "roof-mounted arrays are poorly ventilated; open_rack_glass_glass would be ground-mounted"},
            "dc_ac_ratio": {"value": PV_DC_AC_RATIO, "is_assumption": False,
                            "reason": "common engineering oversizing 1.1-1.3. Implemented as inverter pdc0 = module pdc0 / (ratio * eta_nom). Clips the peak, leaves daily energy essentially unchanged."},
            "gamma_pdc_per_c": {"value": MODULE_GAMMA_PDC, "is_assumption": False,
                                "reason": "standard crystalline silicon temperature coefficient"},
        },
        "limitations": [
            "ERA5 archive is hourly only; 15-minute and finer detail is not available for multi-decade spans.",
            "All feeder nodes fall inside one 25 km ERA5 grid cell, so all PV share the same irradiance; output diversity comes only from array orientation, not from spatial weather variation.",
            "Load comes from an Australian customer and PV from a US site; only the season is matched, not the actual day.",
            ("Synthetic weather: this case was generated by the ARIMA(kt)+copula layer, not replayed; "
             "see output/synth/<site>_validation.csv for the marginal and dynamic errors."
             if weather_mode == "synth" else
             "This case replays observed days; no synthetic generation is applied."),
        ],
        "per_device": extras["per_device"],
    }
    (case / "profile_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _load_weather(site: str, year: int, start: pd.Timestamp, days: int,
                  mode: str = "era5", seed: int = SYNTH_SEED) -> pd.DataFrame:
    """Observed ERA5 for ``year``, or synthetic weather starting at ``start``."""
    if mode == "synth":
        # Imported lazily: era5_synth imports this module for SITES/ERA5_CACHE.
        from data.generate_data import era5_synth
        return era5_synth.sample_weather(site, start, days, seed=seed)
    return _fetch_era5(site, year)


def _single(case_dir: Path, site: str, date: pd.Timestamp, year: int, pv_scale: float = 1.0,
            weather_mode: str = "era5", weather_seed: int = SYNTH_SEED) -> None:
    weather = _load_weather(site, year, date, 1, weather_mode, weather_seed)
    day = weather.loc[weather.index.date == date.date()]
    if len(day) != STEPS_PER_DAY:
        raise SystemExit(f"{date.date()} has {len(day)} hours (expected {STEPS_PER_DAY}). The cache is a fixed-offset "
                          f"clock so this should not happen; check data/raw/era5/{site}_{year}.csv.")
    info = SITES[site]
    tilts, azimuths = _array_geometry(info["lat"], 6)
    pv_pu = np.vstack([_pv_pu(day, info["lat"], info["lon"], info["solar_tz"], tilts[i], azimuths[i]) for i in range(6)])
    wind_pu = _wind_pu(day)
    timestamps = pd.DatetimeIndex(day.index)
    devices, extras = _write_case(case_dir, site, timestamps, pv_pu, wind_pu, EV_SEED, pv_scale=pv_scale)
    _write_meta(case_dir, site, timestamps, tilts, azimuths, extras, weather_mode)
    fleet_pv = sum(extras["per_device"][d["id"]]["peak_kw"] for d in DERS if d["kind"] == "pv")
    pv_installed = sum(d["p_rated_kw"] for d in DERS if d["kind"] == "pv") * pv_scale
    load_peak = pd.read_csv(case_dir / "load_profiles.csv").groupby("timestamp")["p_kw"].sum().max()
    print(f"case              : {case_dir}")
    print(f"site              : {info['label']}")
    print(f"date              : {timestamps[0]} .. {timestamps[-1]} ({STEPS_PER_DAY} x 1 h)")
    print(f"PV peak (fleet)   : {fleet_pv:.1f} kW of {pv_installed:.0f} kW installed")
    print(f"penetration       : {pv_installed / load_peak * 100:.1f}% (PV installed / load peak {load_peak:.0f} kW)")
    print(f"load month used   : {_ausgrid_month_prefix(timestamps[0])} (season-matched)")
    for ident, stats in extras["per_device"].items():
        print(f"  {ident:<14s} peak={stats['peak_kw']:7.2f} kW  max|ramp|={stats['max_ramp_kw_per_min']:6.2f} kW/min")


def _batch(case_dir: Path, site: str, year: int, start: pd.Timestamp, count: int,
           soc_initials: tuple[float, ...], pv_scale: float = 1.0,
           weather_mode: str = "era5", weather_seed: int = SYNTH_SEED) -> None:
    weather = _load_weather(site, year, start, count + 40, weather_mode, weather_seed)
    info = SITES[site]
    tilts, azimuths = _array_geometry(info["lat"], 6)

    # The timeline lives in network.json, and _write_case rewrites it per scenario,
    # so both the NetworkModel and the solver's reference must be refreshed each
    # iteration. Only the topology is invariant, and rebuilding it is cheap.
    solver: OpenDSSSolver | None = None
    rows = []
    checked = 0
    days_done = 0
    output_dir = ROOT / "output" / case_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scenarios").mkdir(exist_ok=True)
    # --scenarios counts CALENDAR DAYS, not rows: one day produces one row per
    # SOC initial value, so counting rows (the previous behaviour) silently
    # divided the requested span by len(soc_initials).
    for offset in range(count + 40):
        if days_done >= count:
            break
        date = start + timedelta(days=offset)
        day = weather.loc[weather.index.date == date.date()]
        checked += 1
        if len(day) != STEPS_PER_DAY:
            continue  # vestigial guard: the fixed-offset cache always has 24 rows/day
        pv_pu = np.vstack([_pv_pu(day, info["lat"], info["lon"], info["solar_tz"], tilts[i], azimuths[i]) for i in range(6)])
        wind_pu = _wind_pu(day)
        timestamps = pd.DatetimeIndex(day.index)
        # The load is rescaled per scenario month, so its peak differs by season:
        # recompute it for each day instead of caching the first day's value,
        # otherwise pv_penetration_pct drifts across a multi-month sweep.
        day_peak = None
        for soc in soc_initials:
            _write_case(case_dir, site, timestamps, pv_pu, wind_pu, EV_SEED, initial_soc=soc, pv_scale=pv_scale)
            if day_peak is None:
                day_peak = float(pd.read_csv(case_dir / "load_profiles.csv").groupby("timestamp")["p_kw"].sum().max())
            network = NetworkModel.from_json(case_dir / "network.json")
            if solver is None:
                solver = OpenDSSSolver(case_dir / "network.dss")
            solver.build(network)
            profiles = ProfileStore.from_case(network, case_dir)
            base = {"date": date.strftime("%Y-%m-%d"), "soc_init_pct": soc, "pv_scale": pv_scale,
                    "pv_penetration_pct": round(550.0 * pv_scale / day_peak * 100, 1),
                    "load_peak_kw": round(day_peak, 1)}
            try:
                _, system, _ = run_qsts(solver, profiles, network)
            except Exception as exc:
                # A single divergent day must NOT abort the whole sweep: at extreme
                # penetration the volt-var loop can fail to converge on some weather
                # days. Record it as status=diverged and keep going; filter it out
                # (status == "ok") when analysing the distribution.
                rows.append({**base, "status": f"diverged:{type(exc).__name__}"})
                continue
            # Per-scenario detail for drill-down (system-level time series only).
            detail = output_dir / "scenarios" / f"{date:%Y-%m-%d}_soc{int(soc)}"
            detail.mkdir(parents=True, exist_ok=True)
            system.to_csv(detail / "qsts_system.csv", index=False)
            pv_only = system[[c for c in system.columns if c.startswith("pv_") and c.endswith("_p_kw")]].sum(axis=1)
            wind_only = system[[c for c in system.columns if c.startswith("wind_") and c.endswith("_p_kw")]].sum(axis=1)
            fleet = pv_only + wind_only
            # Net load must be the SUBSTATION exchange, not "load minus PV": only
            # source_p_kw accounts for BESS, EV and losses. OpenDSS reports the
            # source as negative while the feeder imports, so net = -source_p_kw
            # (positive = import, negative = export/reverse flow).
            net = -system["source_p_kw"]
            # NOTE: max_v_pu is NOT a useful randomness metric at this penetration --
            # it is pinned at the regulator tap ceiling (1.1 pu) whenever the evening
            # peak pushes the regulators to their limit, so it barely responds to the
            # weather. Use min_v_pu and the net-load ramp instead.
            # The grill-defined indicators need DISTRIBUTIONS, not just one extreme
            # value per scenario, so record percentiles and event counts here:
            #   - intermittency: hour-to-hour PV drops larger than 20% of installed
            #   - variability : P95 net-load ramp (P50/P95/P99 across scenarios later)
            #   - duck curve  : evening net peak minus midday net trough
            pv_step = np.diff(pv_only.to_numpy(float))
            net_step = np.diff(net.to_numpy(float))
            hours = np.asarray(timestamps.hour)
            # The duck curve is "evening peak minus MIDDAY trough". Using the global
            # net minimum instead is wrong: that minimum sits at the pre-dawn load
            # valley (05:00), not in the PV hours, which made the depth *decrease*
            # with penetration -- the opposite of the real effect.
            midday = (hours >= 9) & (hours <= 16)
            evening = hours >= 17
            net_arr = net.to_numpy(float)
            drop_floor = 0.20 * 550.0 * pv_scale
            rows.append({
                **base, "status": "ok",
                # pv_scale=0 is the legitimate "no DER" baseline (L0), so guard the
                # capacity-factor denominators instead of emitting nan/inf.
                "pv_capacity_factor": round(float(pv_only.mean() / (550.0 * pv_scale)), 4) if pv_scale else 0.0,
                "wind_capacity_factor": round(float(wind_only.mean() / 120.0), 4),
                "der_energy_share_pct": round(float(fleet.sum() / system["total_load_kw"].sum() * 100), 2),
                "net_std_kw": round(float(net.std()), 1),
                "net_ramp_max_kw_per_min": round(float(np.abs(net_step).max()) / 60.0, 2),
                "net_ramp_p95_kw_per_min": round(float(np.percentile(np.abs(net_step), 95)) / 60.0, 2),
                "pv_drop_events": int((pv_step < -drop_floor).sum()) if pv_scale else 0,
                "pv_drop_max_kw": round(float(-pv_step.min()) if pv_scale else 0.0, 1),
                "net_min_kw": round(float(net_arr.min()), 1),
                "net_midday_min_kw": round(float(net_arr[midday].min()), 1),
                "net_evening_max_kw": round(float(net_arr[evening].max()), 1),
                "duck_depth_kw": round(float(net_arr[evening].max() - net_arr[midday].min()), 1),
                "min_v_pu": round(float(system["min_v_pu"].min()), 4),
                "max_v_pu": round(float(system["max_v_pu"].max()), 4),
                "v_high_steps": int(system["voltage_high_count"].sum()),
                "v_low_steps": int(system["voltage_low_count"].sum()),
                "reverse_steps": int(system["reverse_power_flow"].sum()),
                "max_line_loading_pct": round(float(system["max_line_loading_pct"].max()), 1),
            })
        days_done += 1

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "scenarios.csv", index=False)
    ok = frame[frame["status"] == "ok"]
    diverged = int((frame["status"] != "ok").sum())
    print(f"scenario days     : {days_done} of {count} requested "
          f"(skipped {checked - days_done} incomplete days)")
    print(f"rows              : {len(frame)} = {days_done} days x {len(soc_initials)} SOC value(s) "
          f"(ok {len(ok)}, diverged {diverged})")
    print(f"written           : {output_dir / 'scenarios.csv'}")
    print(f"per-scenario detail: {output_dir / 'scenarios'}/<date>_soc<soc>/qsts_system.csv")
    print()
    print(ok.groupby("soc_init_pct")[["pv_capacity_factor", "wind_capacity_factor",
                                      "der_energy_share_pct", "net_std_kw", "net_ramp_max_kw_per_min",
                                      "net_ramp_p95_kw_per_min", "pv_drop_events", "duck_depth_kw",
                                      "min_v_pu", "max_v_pu", "max_line_loading_pct",
                                      "reverse_steps"]].mean().T.to_string())
    print()
    print("跨场景离散度（体现随机性的是这些量的分布，而非单场景数值）：")
    spread = ok[["pv_capacity_factor", "net_std_kw", "net_ramp_max_kw_per_min",
                 "duck_depth_kw", "min_v_pu"]].agg(["mean", "std", "min", "max"])
    print(spread.round(4).to_string())


# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate an IEEE-13 case driven by ERA5 weather through pvlib.")
    parser.add_argument("--site", default="atlanta", choices=sorted(SITES))
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--date", default=None, help="Single-day mode: YYYY-MM-DD.")
    parser.add_argument("--start", default="2022-01-01", help="Batch mode start date.")
    parser.add_argument("--scenarios", type=int, default=None, help="Batch mode: number of days to run.")
    parser.add_argument("--soc-init", default="55", help="Comma-separated BESS initial SOC values, e.g. 20,50,80.")
    parser.add_argument("--pv-scale", type=float, default=1.0,
                        help="Multiply every PV p_rated_kw, for a penetration sweep. Penetration is PV installed / "
                             "measured load peak (~5025 kW), so 1.0 = 550 kW = 10.9%%; 3.6 -> ~1980 kW (39%%), "
                             "6.9 -> ~3795 kW (75%%), 9.1 -> 100%%. Reverse power flow appears from ~18; the volt-var "
                             "loop still converges to ~22 at relaxation 0.2.")
    parser.add_argument("--name", default=DEFAULT_CASE, help="Output case directory name.")
    parser.add_argument("--weather", default="era5", choices=("era5", "synth"),
                        help="era5 = replay observed days; synth = ARIMA(kt)+copula generated weather.")
    parser.add_argument("--synth-seed", type=int, default=SYNTH_SEED,
                        help="RNG seed for --weather synth (same seed -> same weather).")
    args = parser.parse_args(argv)

    case_dir = ROOT / "data" / args.name

    if args.scenarios:
        soc = tuple(float(v) for v in args.soc_init.split(","))
        start = pd.Timestamp(args.start)
        if start.year != args.year:
            # _batch fetches exactly one calendar year of weather and then matches
            # days inside it, so a start date outside that year would silently
            # produce zero scenarios. Same rule as single-day mode: the date wins.
            print(f"note              : --year {args.year} overridden by --start {start.date()} -> {start.year}")
            args.year = start.year
        _ensure_case(case_dir, start)
        _batch(case_dir, args.site, args.year, start, args.scenarios, soc, args.pv_scale,
               args.weather, args.synth_seed)
        return

    date = pd.Timestamp(args.date) if args.date else pd.Timestamp(f"{args.year}-09-20")
    if date.year != args.year:
        args.year = date.year
    _single(case_dir, args.site, date, args.year, pv_scale=args.pv_scale,
            weather_mode=args.weather, weather_seed=args.synth_seed)


if __name__ == "__main__":
    _main()
