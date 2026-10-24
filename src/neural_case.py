"""Convert a saved neural weather member into a standard runnable QSTS case.

Only unstandardized transformed fields (archive key ``raw``) are interpolated.
The neural latent state must never be interpreted as physical weather. There is
no retraining, Kalman-model dependency or additional local stochastic field here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.generate_data import ieee13_era5 as base
from src.regional_spde import FIELDS
from src.site_weather import read_layout, weather_from_raw, site_power
from src.spde_case import export_case
from src.weather_spde import RectGrid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_member(weather_path, prepared, *, member=0, start_index=0, hours=None):
    """Read one archived hourly trajectory with its exact ERA5 node ordering."""
    weather_path, prepared = Path(weather_path), Path(prepared)
    manifest_path = prepared/"manifest.json"
    sites_path = prepared/"sites.csv"
    manifest_sha = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest["field_order"]) != FIELDS:
        raise ValueError("Prepared field order differs from the neural weather contract.")
    source_sites = pd.read_csv(sites_path)
    if source_sites.site_id.duplicated().any():
        raise ValueError("Source site IDs must be unique.")
    with np.load(weather_path, allow_pickle=False) as archive:
        required = {"raw", "timestamps", "site_ids", "field_order", "grid_x_m", "grid_y_m", "source_manifest_sha256"}
        if not required.issubset(archive.files):
            raise ValueError("Neural archive must contain unstandardized raw fields and source/grid metadata.")
        if str(archive["source_manifest_sha256"].item()) != manifest_sha:
            raise ValueError("Neural weather and prepared manifest have different provenance.")
        if tuple(archive["field_order"].tolist()) != FIELDS:
            raise ValueError("Neural field order must be solar_logit, wind_u_100m, wind_v_100m, temp_air.")
        node_ids = archive["site_ids"].astype(str)
        if len(np.unique(node_ids)) != len(node_ids) or set(node_ids) != set(source_sites.site_id):
            raise ValueError("Neural and prepared node IDs must match exactly.")
        source_sites = source_sites.set_index("site_id").loc[node_ids].reset_index()
        x, y = archive["grid_x_m"].copy(), archive["grid_y_m"].copy()
        times = pd.to_datetime(archive["timestamps"], unit="ns", utc=True)
        if len(times) < 2 or not np.all(np.diff(times.asi8) == 3600*10**9):
            raise ValueError("Neural weather must contain consecutive hourly midpoint timestamps.")
        if not ((times.minute == 30) & (times.second == 0) & (times.microsecond == 0) & (times.nanosecond == 0)).all():
            raise ValueError("Neural hourly weather timestamps must label UTC interval midpoints.")
        raw = archive["raw"]
        if raw.ndim != 4 or raw.shape[1:] != (len(times), 4, len(node_ids)):
            raise ValueError("Raw archive shape must be member,time,4,node.")
        if not 0 <= member < len(raw):
            raise ValueError(f"Member must be between 0 and {len(raw)-1}.")
        stop = len(times) if hours is None else start_index+hours
        if start_index < 0 or stop > len(times) or stop-start_index < 2:
            raise ValueError("Select at least two available hourly steps with --start-index/--hours.")
        selected = raw[member:member+1, start_index:stop].astype(float)
        if not np.isfinite(selected).all():
            raise ValueError("Selected neural raw weather must be finite.")
        times = times[start_index:stop]
        members = len(raw)
    # Canonical y,x order is required by the bilinear observation matrix.
    order = np.lexsort((source_sites.x_m, source_sites.y_m))
    source_sites = source_sites.iloc[order].reset_index(drop=True)
    grid = RectGrid(x, y)
    if len(source_sites) != grid.size or not np.allclose(source_sites[["x_m", "y_m"]], grid.xy):
        raise ValueError("Source coordinates do not form the archived rectangular grid.")
    selected = selected[..., order]
    metadata = {"weather_source": "saved neural-coefficient SPDE trajectory",
        "neural_weather_sha256": digest(weather_path), "source_manifest_sha256": manifest_sha,
        "source_sites_sha256": digest(sites_path), "member": member, "available_members": members,
        "start_index": start_index, "hours": len(times), "field_order": list(FIELDS),
        "time_start_midpoint_utc": str(times[0]), "time_end_midpoint_utc": str(times[-1])}
    return selected, times, source_sites, grid, metadata


def interpolate_member(raw, source_sites, grid, sites):
    """Bilinear raw-field interpolation; derive source projection from its grid.

    The station-layout origin is independent of the regional grid origin. Do not
    apply station-local east/north offsets directly to regional x/y coordinates.
    """
    longitude = source_sites.longitude.to_numpy().reshape(len(grid.y), len(grid.x))
    latitude = source_sites.latitude.to_numpy().reshape(len(grid.y), len(grid.x))
    if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
        raise ValueError("Source geographic coordinates must be finite.")
    if not np.allclose(longitude, longitude[0:1]) or not np.allclose(latitude, latitude[:, 0:1]):
        raise ValueError("Expected a separable latitude/longitude regional grid.")
    lon_axis, lat_axis = longitude[0], latitude[:, 0]
    if (np.diff(lon_axis) <= 0).any() or (np.diff(lat_axis) <= 0).any():
        raise ValueError("Source geographic axes must increase.")
    ll = sites[["longitude", "latitude"]].to_numpy(float)
    if (not np.isfinite(ll).all() or (ll[:, 0] < lon_axis[0]).any() or (ll[:, 0] > lon_axis[-1]).any()
            or (ll[:, 1] < lat_axis[0]).any() or (ll[:, 1] > lat_axis[-1]).any()):
        raise ValueError("A station lies outside the neural weather domain; extrapolation is not supported.")
    xy = np.column_stack((np.interp(ll[:, 0], lon_axis, grid.x), np.interp(ll[:, 1], lat_axis, grid.y)))
    weights = grid.observation_matrix(xy).toarray()
    values = np.einsum("mtfn,sn->mtfs", raw, weights)
    nearest = np.argmin(np.sum((xy[:, None]-grid.xy[None])**2, axis=-1), axis=1)
    return values, {"regional_xy_m": xy.tolist(), "bilinear_weights": weights.tolist(),
                    "source_node_ids": source_sites.site_id.tolist(),
                    "nearest_regional_node": nearest.tolist()}


def build_case(weather_path, prepared, layout_path, physics_path, case, *, member=0, start_index=0, hours=None):
    """Export existing neural weather with assumed station equipment; no new noise."""
    case = Path(case)
    if case.exists() and any(case.iterdir()):
        raise FileExistsError(f"Case already contains files: {case}; use a new --case.")
    layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))
    physics = json.loads(Path(physics_path).read_text(encoding="utf-8"))
    sites = read_layout(layout)
    raw, times, source_sites, grid, metadata = read_member(weather_path, prepared,
        member=member, start_index=start_index, hours=hours)
    raw, interpolation = interpolate_member(raw, source_sites, grid, sites)
    weather = weather_from_raw(raw, times, sites, physics["pv_wind_10m_shear_exponent"])
    power, decomposition = site_power(weather, times, sites, physics)
    templates = {d["id"]: d for d in base._ders()}
    devices, frames, prescribed = [], {}, {}
    starts = times-pd.Timedelta(minutes=30)
    for i, row in enumerate(sites.to_dict("records")):
        template = templates.get(row["template_id"])
        if template is None or template["kind"] != row["kind"]:
            raise ValueError("Station template must identify a compatible existing electrical device.")
        device = dict(template)
        device.update(id=row["device_id"], p_rated_kw=row["p_rated_kw"],
            s_rated_kva=base.INVERTER_S_RATIO*row["p_rated_kw"], latitude=row["latitude"],
            longitude=row["longitude"], weather_site_id=row["site_id"],
            profile_file=f"der_profiles/{row['device_id'].lower()}.csv")
        for key in ("bus", "surface_tilt", "surface_azimuth", "hub_height_m", "shear_exponent",
                    "cut_in_m_s", "rated_speed_m_s", "cut_out_m_s"):
            if key in row and pd.notna(row[key]):
                device[key] = row[key]
        devices.append(device)
        frame = pd.DataFrame({key: weather[key][0, :, i] for key in
            ("ghi", "temp_air", "wind_speed", "wind_speed_100m", "wind_u_100m", "wind_v_100m")}, index=starts)
        radiation = decomposition.get(row["device_id"])
        frame["dni"] = radiation["dni"][0] if radiation else 0.
        frame["dhi"] = radiation["dhi"][0] if radiation else frame.ghi
        frames[row["device_id"]] = frame
        prescribed[row["device_id"]] = power[0, :, i]
    metadata.update(status="neural weather to QSTS; example station layout, not station-calibrated weather",
        coordinate_origin=layout["coordinate_origin"],
        coordinates_are_assumed=layout.get("coordinates_are_assumed", True),
        equipment_parameters_are_assumed=layout.get("equipment_parameters_are_assumed", True),
        local_residual_added=False, site_configuration=layout, regional_interpolation=interpolation,
        layout_sha256=digest(layout_path), physics_configuration_sha256=digest(physics_path),
        power_time_convention="interval start UTC; physics at hourly midpoint",
        loads="original synthetic reference daily shapes repeated across days; not weather-conditioned",
        clear_sky_altitude_assumption_m=100.,
        weather_geography_source="neural raw transformed fields bilinearly interpolated to configured stations",
        pv_total_kw=float(sites.loc[sites.kind == "pv", "p_rated_kw"].sum()),
        wind_total_kw=float(sites.loc[sites.kind == "wind", "p_rated_kw"].sum()))
    export_case(case, starts, frames, sites, devices, metadata, active_power_kw=prescribed)
    pd.concat([frame.assign(device_id=name).rename_axis("timestamp_utc").reset_index()
               for name, frame in frames.items()], ignore_index=True).to_csv(case/"weather_by_site.csv", index=False)
    return metadata
