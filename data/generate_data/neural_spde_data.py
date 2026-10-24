"""Download multi-year ERA5 on the existing 6x6 grid, in resumable quarters."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.regional_weather import fetch_region, prepare_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--output", type=Path, default=ROOT/"data/processed/spde/multiyear_2015_2019")
    args = parser.parse_args()
    if args.end_year-args.start_year < 2:
        parser.error("Use >=3 consecutive years for disjoint training/validation/test years.")
    args.output.mkdir(parents=True, exist_ok=True)
    source_root = ROOT/"data/raw/era5_neural"
    manifest_path = args.output/"manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "source": "ERA5 only via Open-Meteo; CC BY 4.0",
        "documentation": "https://open-meteo.com/en/docs/historical-weather-api",
        "scope": "regional hourly reanalysis, no fine-scale station truth",
        "years": {}, "field_order": ["solar_logit", "wind_u_100m", "wind_v_100m", "temp_air"],
        "split": {"train": list(range(args.start_year, args.end_year-1)),
                  "validation": [args.end_year-1], "test": [args.end_year]},
        "held_sites": [14, 15, 20, 21], "solar_cap": 1.6}
    expected_split = {"train": list(range(args.start_year, args.end_year-1)),
                      "validation": [args.end_year-1], "test": [args.end_year]}
    if manifest["split"] != expected_split:
        raise ValueError("Existing prepared split differs; choose a new --output.")
    for year in range(args.start_year, args.end_year+1):
        arrays, records, sites_ref = [], [], None
        destination = args.output/f"era5_{year}.npz"
        if destination.exists() and str(year) in manifest["years"]:
            if hashlib.sha256(destination.read_bytes()).hexdigest() != manifest["years"][str(year)]["sha256"]:
                raise ValueError("Prepared year checksum mismatch.")
            print(f"Cached year {year}", flush=True)
            continue
        for month in (1, 4, 7, 10):
            begin = pd.Timestamp(year, month, 1)
            end = begin+pd.offsets.QuarterEnd()
            print(f"ERA5 {begin.date()} to {end.date()}: 36 distinct cells", flush=True)
            for attempt in range(5):
                try:
                    raw, url = fetch_region(source_root, str(begin.date()), str(end.date()),
                        latitudes=[50.25, 50.5, 50.75, 51., 51.25, 51.5],
                        longitudes=[5.75, 6., 6.25, 6.5, 6.75, 7.])
                    frame, sites, provenance = prepare_region(raw, url, str(begin.date()), str(end.date()))
                    break
                except Exception as error:
                    print(f"  attempt {attempt+1}: {type(error).__name__}: {error}", flush=True)
                    if attempt == 4:
                        raise
                    time.sleep(min(10*2**attempt, 60))
            if sites_ref is not None:
                pd.testing.assert_frame_equal(sites, sites_ref)
            sites_ref = sites
            times = pd.DatetimeIndex(sorted(frame.timestamp_utc.unique())).as_unit("ns")
            def values(name):
                return frame.pivot(index="timestamp_utc", columns="site_id", values=name).reindex(
                    index=times, columns=sites.site_id).to_numpy(np.float32)
            kc = values("clear_sky_index")
            p = np.clip(kc/1.6, .001, .999)
            raw_fields = np.stack([np.log(p/(1-p)), values("wind_u_100m"), values("wind_v_100m"), values("temp_air")], axis=1)
            arrays.append({"raw": raw_fields, "clear": values("clear_sky_ghi"),
                           "ghi": values("ghi"), "times": times.asi8})
            records.append(provenance)
            del frame
            time.sleep(2)
        combined = {key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
        np.savez_compressed(destination, **combined)
        sites_ref.to_csv(args.output/"sites.csv", index=False)
        manifest["years"][str(year)] = {"file": destination.name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "hours": len(combined["times"]), "quarters": records}
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        print(f"Prepared {year}: {len(combined['times'])} hours, {destination.stat().st_size/1e6:.1f} MB", flush=True)
    print(f"Complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
