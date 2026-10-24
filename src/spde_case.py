"""Export site-specific weather and power into the existing QSTS case format."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from data.generate_data import ieee13_era5 as base
from src.power_flow import NetworkModel
from src.profile_store import ProfileStore

ROOT = Path(__file__).resolve().parents[1]


def export_case(case: Path, timestamps: pd.DatetimeIndex, weather: dict[str, pd.DataFrame],
                sites: pd.DataFrame, devices: list[dict], metadata: dict, *,
                active_power_kw: dict[str, np.ndarray] | None = None) -> None:
    """Write an isolated case; refuse to overwrite a populated directory."""
    if case.exists() and any(case.iterdir()):
        raise FileExistsError(f"Output case already contains files: {case}. Choose a new --name.")
    if timestamps.tz is None or len(timestamps) < 2:
        raise ValueError("At least two timezone-aware, regularly spaced timestamps are required.")
    dt = np.diff(timestamps.asi8) / 1e9
    if not np.all(dt == dt[0]) or dt[0] <= 0 or dt[0] != int(dt[0]):
        raise ValueError("Time grid must have a positive, whole-second constant step.")
    if active_power_kw is not None:
        if set(active_power_kw) != {device["id"] for device in devices}:
            raise ValueError("Precomputed powers must match every DER device exactly.")
        for device in devices:
            values = np.asarray(active_power_kw[device["id"]], float)
            if (values.shape != (len(timestamps),) or not np.isfinite(values).all()
                    or (values < 0).any() or (values > device["p_rated_kw"]+1e-8).any()):
                raise ValueError("Precomputed power must be finite, nonnegative and within each device rating.")
    step = int(dt[0])
    case.mkdir(parents=True, exist_ok=True)
    (case/"der_profiles").mkdir()
    network = base._network(devices)
    utc = timestamps.tz_convert("UTC").tz_localize(None)
    network["simulation"] = {"start_datetime": str(utc[0]),
                             "end_datetime": str(utc[-1]+np.timedelta64(step, "s")), "step_seconds": step}
    network["weather_geography"] = {"origin": metadata["coordinate_origin"],
                                     "source": metadata.get("weather_geography_source", "assumed mechanism-demo layout")}
    (case/"network.json").write_text(json.dumps(network, indent=2)+"\n", encoding="utf-8")
    # Reuse an existing reference DSS rather than requiring another download.
    for filename in ("IEEE13Nodeckt.dss", "IEEELineCodes.DSS"):
        sources = sorted((ROOT/"data").glob("*/ieee13_base/"+filename))
        if sources:
            (case/"ieee13_base").mkdir(exist_ok=True)
            shutil.copyfile(sources[0], case/"ieee13_base"/filename)
    base._write_dss(case, devices)
    (case/"rms_config.json").write_text(json.dumps(base.RMS_CONFIG, indent=2)+"\n", encoding="utf-8")
    stamp = utc.strftime("%Y-%m-%d %H:%M:%S")
    hours = utc.hour.to_numpy()+utc.minute.to_numpy()/60+utc.second.to_numpy()/3600
    loads = []
    for ident, (p, q) in base._reference_load_curves().items():
        loads.append(pd.DataFrame({"timestamp": stamp, "load_name": ident,
                                   "p_kw": np.interp(hours, np.arange(25), np.r_[p, p[0]]),
                                   "q_kvar": np.interp(hours, np.arange(25), np.r_[q, q[0]])}))
    pd.concat(loads, ignore_index=True).to_csv(case/"load_profiles.csv", index=False)
    power = {}
    for device in devices:
        row = sites.set_index("device_id").loc[device["id"]]
        w = weather[device["id"]]
        # Physics is evaluated at the interval midpoint; output labels stay at interval starts.
        physics = w.copy()
        physics.index = physics.index+np.timedelta64(step*500_000_000, "ns")
        rating = float(device["p_rated_kw"])
        if device["kind"] == "pv":
            p = (np.asarray(active_power_kw[device["id"]], float) if active_power_kw is not None else
                 rating*base._pv_pu(physics, row.latitude, row.longitude, "UTC", row.latitude, 180.))
            q_limit = np.sqrt(np.maximum(device["s_rated_kva"]**2-p**2, 0))
            frame = pd.DataFrame({"timestamp": stamp, "pv_kw": p, "pv_kvar": 0.,
                                  "controller_type": "volt_var", "q_limit_kvar": q_limit, "v_ref_pu": 1.,
                                  "droop_kvar_per_pu": rating*base.VOLT_VAR_DROOP_PER_RATING,
                                  "deadband_pu": base.VOLT_VAR_DEADBAND_PU})
        else:
            p = (np.asarray(active_power_kw[device["id"]], float) if active_power_kw is not None else
                 rating*base._wind_pu(physics))
            frame = pd.DataFrame({"timestamp": stamp, "wind_kw": p, "wind_kvar": 0., "controller_type": "constant_pq"})
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Power conversion produced invalid values.")
        frame.to_csv(case/device["profile_file"], index=False)
        power[device["id"]] = p
    pd.DataFrame(power, index=timestamps).rename_axis("timestamp_utc").to_csv(case/"power_by_site.csv")
    sites.to_csv(case/"sites.csv", index=False)
    (case/"profile_meta.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    # Exercise the actual downstream contract rather than only checking CSV columns.
    net = NetworkModel.from_json(case/"network.json")
    ProfileStore.from_case(net, case)
