"""Fetch many years of ERA5 for one site into data/raw/era5/ (cache format used by ieee13_era5.py).

Open-Meteo's archive endpoint accepts a multi-year range in a single request (verified:
1995-2000 = 43824 rows in ~30 s), so years are pulled in chunks instead of one call per
year (one call per year costs ~5 min and times out roughly one time in five).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.generate_data.ieee13_era5 import ERA5_CACHE, HOURLY, SITES, WEATHER_API  # noqa: E402


def fetch_chunk(site: str, y0: int, y1: int, retries: int = 3) -> pd.DataFrame:
    info = SITES[site]
    url = (
        f"{WEATHER_API}?latitude={info['lat']}&longitude={info['lon']}"
        f"&start_date={y0}-01-01&end_date={y1}-12-31"
        f"&hourly={HOURLY}&timezone={urllib.parse.quote(info['tz'])}&wind_speed_unit=ms"
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=900) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit(f"{site} {y0}-{y1} failed after {retries} tries: {last}")

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
        raise SystemExit(f"ERA5 response contains missing values for {site} {y0}-{y1}.")
    return frame


def main() -> None:
    site = sys.argv[1]
    y0, y1 = int(sys.argv[2]), int(sys.argv[3])
    chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    ERA5_CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    start = y0
    while start <= y1:
        stop = min(start + chunk - 1, y1)
        have = [(ERA5_CACHE / f"{site}_{y}.csv").exists() for y in range(start, stop + 1)]
        if all(have):
            print(f"{site} {start}-{stop}: cached", flush=True)
            start = stop + 1
            continue
        frame = fetch_chunk(site, start, stop)
        for year, part in frame.groupby(frame.index.year):
            path = ERA5_CACHE / f"{site}_{year}.csv"
            if path.exists():
                continue
            part.to_csv(path)
        print(f"{site} {start}-{stop}: {len(frame)} rows -> {stop - start + 1} files "
              f"({time.time() - t0:.0f}s)", flush=True)
        start = stop + 1
    print(f"{site} done {y0}-{y1} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
