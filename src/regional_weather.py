"""Same-product ERA5 weather on distinct neighboring cells, in common UTC hours."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.parse

import numpy as np
import pandas as pd
from pvlib.location import Location

from src.weather_io import download_https

API = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
             "temperature_2m", "wind_speed_10m", "wind_speed_100m", "wind_direction_100m")


def wind_components(speed: np.ndarray, direction_degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological direction is where wind comes FROM: eastward u, northward v."""
    angle = np.deg2rad(direction_degrees)
    return -np.asarray(speed)*np.sin(angle), -np.asarray(speed)*np.cos(angle)


def fetch_region(cache: Path, start: str, end: str, *, latitudes: list[float],
                 longitudes: list[float]) -> tuple[Path, str]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts.tz is not None or end_ts.tz is not None or end_ts < start_ts:
        raise ValueError("Use timezone-free start/end calendar dates in ascending order.")
    pairs = [(lat, lon) for lat in latitudes for lon in longitudes]
    if len(pairs) < 4 or len(set(pairs)) != len(pairs):
        raise ValueError("Request distinct coordinates on at least a 2x2 grid.")
    query = {"latitude": ",".join(str(p[0]) for p in pairs),
             "longitude": ",".join(str(p[1]) for p in pairs),
             "start_date": start_ts.strftime("%Y-%m-%d"),
             # One extra day supplies the endpoint of the final mean interval.
             "end_date": (end_ts+pd.offsets.Day()).strftime("%Y-%m-%d"),
             "hourly": ",".join(VARIABLES), "models": "era5", "timezone": "UTC",
             "wind_speed_unit": "ms", "elevation": ",".join(["nan"]*len(pairs)),
             "cell_selection": "nearest"}
    url = API+"?"+urllib.parse.urlencode(query)
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache/f"region_{start_ts:%Y%m%d}_{end_ts:%Y%m%d}_{key}.json"
    if not path.exists():
        download_https(url, path)
    return path, url


def prepare_region(path: Path, url: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("Expected a multi-location response from the ERA5-only request.")
    used, frames, stations, times = set(), [], [], None
    start_utc = pd.Timestamp(start, tz="UTC")
    stop_utc = pd.Timestamp(end, tz="UTC")+pd.offsets.Day()
    for i, point in enumerate(payload):
        if point.get("utc_offset_seconds") != 0:
            raise ValueError("Regional weather must be returned in UTC.")
        pair = (float(point["latitude"]), float(point["longitude"]))
        if pair in used:
            raise ValueError("Multiple requests resolved to the same ERA5 cell; do not treat them as separate sites.")
        used.add(pair)
        units = point["hourly_units"]
        if units.get("wind_speed_100m") != "m/s" or units.get("wind_speed_10m") != "m/s":
            raise ValueError("Wind speeds must be m/s.")
        raw = pd.DataFrame(point["hourly"])
        index = pd.DatetimeIndex(pd.to_datetime(raw.pop("time"), utc=True))
        if index.has_duplicates or not np.all(np.diff(index.asi8) == 3_600_000_000_000):
            raise ValueError("ERA5 response is not a unique, regular hourly series.")
        if times is not None and not times.equals(index):
            raise ValueError("Weather cells do not share the same timestamps.")
        times = index
        values = raw[list(VARIABLES)].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Missing weather in response; do not silently impute targets.")
        u, v = wind_components(raw.wind_speed_100m.to_numpy(), raw.wind_direction_100m.to_numpy())
        # Radiation at endpoint t describes [t-1h,t]. Instantaneous temperature
        # and wind are linearly averaged between the same two endpoints.
        middle = index[1:]-np.timedelta64(30, "m")
        frame = pd.DataFrame({"timestamp_utc": middle,
                              "ghi": raw.shortwave_radiation.to_numpy()[1:],
                              "dni": raw.direct_normal_irradiance.to_numpy()[1:],
                              "dhi": raw.diffuse_radiation.to_numpy()[1:],
                              "temp_air": .5*(raw.temperature_2m.to_numpy()[1:]+raw.temperature_2m.to_numpy()[:-1]),
                              "wind_u_100m": .5*(u[1:]+u[:-1]), "wind_v_100m": .5*(v[1:]+v[:-1]),
                              "wind_speed": .5*(raw.wind_speed_10m.to_numpy()[1:]+raw.wind_speed_10m.to_numpy()[:-1])})
        frame["wind_speed_100m"] = np.hypot(frame.wind_u_100m, frame.wind_v_100m)
        frame = frame[(frame.timestamp_utc >= start_utc) & (frame.timestamp_utc < stop_utc)].copy()
        expected = pd.date_range(start_utc+pd.Timedelta(minutes=30), stop_utc, freq="h", inclusive="left")
        if not pd.DatetimeIndex(frame.timestamp_utc).equals(expected):
            raise ValueError("Response does not cover every requested hourly interval.")
        site_id = f"ERA5_{i:03d}"
        frame["site_id"] = site_id
        location = Location(*pair, tz="UTC", altitude=float(point["elevation"]))
        center = pd.DatetimeIndex(frame.timestamp_utc)
        clear = np.zeros(len(frame))
        for offset, weight in zip((-30, -15, 0, 15, 30), (1, 4, 2, 4, 1), strict=True):
            clear += weight*location.get_clearsky(center+np.timedelta64(offset, "m"))["ghi"].to_numpy()/12
        frame["clear_sky_ghi"] = clear
        frame["clear_sky_index"] = np.where(clear >= 50, frame.ghi/np.maximum(clear, 1), np.nan)
        frames.append(frame)
        stations.append({"site_id": site_id, "latitude": pair[0], "longitude": pair[1],
                         "elevation_m": float(point["elevation"])})
    sites = pd.DataFrame(stations).sort_values(["latitude", "longitude"]).reset_index(drop=True)
    lat0, lon0 = sites[["latitude", "longitude"]].mean()
    sites["x_m"] = np.deg2rad(sites.longitude-lon0)*6_371_000*np.cos(np.deg2rad(lat0))
    sites["y_m"] = np.deg2rad(sites.latitude-lat0)*6_371_000
    if len(sites.latitude.unique())*len(sites.longitude.unique()) != len(sites):
        raise ValueError("Resolved cells are not a rectangular grid.")
    metadata = {"source": "ERA5 reanalysis via Open-Meteo, explicit models=era5", "request_url": url,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_file": str(path.resolve()), "start": str(start_utc), "end_exclusive": str(stop_utc),
                "coordinate_origin": {"latitude": float(lat0), "longitude": float(lon0)},
                "coordinates": "returned cell centres, not requested points; local equirectangular metres",
                "downscaling": "disabled using elevation=nan for every coordinate",
                "hourly_intervals": len(expected), "distinct_cells": len(sites),
                "time_alignment": "midpoint UTC labels; radiation preceding-hour mean, T/u/v averaged endpoint instants",
                "wind_convention": "meteorological from-direction, u=-speed*sin(dir), v=-speed*cos(dir)",
                "weather_is_observed_station_data": False,
                "scope": "regional hourly weather, not feeder-scale minute cloud/turbulence validation",
                "documentation": "https://open-meteo.com/en/docs/historical-weather-api"}
    return pd.concat(frames, ignore_index=True), sites, metadata
