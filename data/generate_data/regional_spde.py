"""Prepare, calibrate and validate same-region hourly solar/wind/temperature SPDEs.

Run with the project's venv Python. Prepare downloads public ERA5-only data;
fit freezes training parameters before validation selection and final testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.regional_weather import fetch_region, prepare_region
from src.regional_spde import (FIELDS, DT, load_region, fit_field, filtered_driver_correlation,
                               system, evaluate, sample_contiguous, regional_driver_model)
from src.weather_spde import (FieldParameters, JointWeatherSPDE, kalman_update,
                              implementation_fingerprint)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def validation_nll(data, field, fit):
    model = JointWeatherSPDE(data.grid, [FieldParameters(**fit["parameters"])], DT)
    f, g, covariance = model.dense_single_field()
    q = g@g.T
    mean = np.zeros(data.grid.size)
    h = np.eye(data.grid.size)[data.train_sites]
    nll = 0.
    for t in range(data.train_end-48, data.validation_end):
        mean, covariance = f@mean, f@covariance@f.T+q
        mean, covariance, loss = kalman_update(mean, covariance, data.z[t, field, data.train_sites],
                                              h, fit["observation_std"])
        if t >= data.train_end:
            nll += loss
    return float(nll)


def fit_experiment(args):
    data = load_region(args.prepared, args.train_days, args.validation_days)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    context = {"data_sha256": hashlib.sha256((args.prepared/"weather_hourly.csv.gz").read_bytes()).hexdigest(),
               "train_end": data.train_end, "validation_end": data.validation_end,
               "train_sites": data.train_sites.tolist(), "fit_budget": args.fit_budget}
    fits, all_fits = [], {}
    for i, name in enumerate(FIELDS):
        candidates = {}
        for mode in ("spatial_ou", "transport_spde"):
            path = out/f"fit_{name}_{mode}.json"
            if args.resume and path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["context"] != context:
                    raise ValueError("Checkpoint data/split/budget differs. Use another output directory.")
                fit = saved["fit"]
                print(f"Reusing {name}/{mode}", flush=True)
                if not fit["optimizer_converged"]:
                    print("  Continuing unfinished training optimization (no test-score tuning)", flush=True)
                    previous = fit
                    fit = fit_field(data, i, transport=mode == "transport_spde",
                                    max_evaluations=args.fit_budget, warm_start=previous)
                    fit["previous_optimization"] = {"nll": previous["training_composite_nll"],
                                                    "evaluations": previous["evaluations"],
                                                    "message": previous["optimizer_message"]}
                    write_json(path, {"context": context, "fit": fit, "implementation": implementation_fingerprint()})
            else:
                print(f"Fitting {name}/{mode}, training cells={len(data.train_sites)}", flush=True)
                fit = fit_field(data, i, transport=mode == "transport_spde",
                                max_evaluations=args.fit_budget, warm_start=candidates.get("spatial_ou"))
                write_json(path, {"context": context, "fit": fit, "implementation": implementation_fingerprint()})
            fit["validation_one_step_nll"] = validation_nll(data, i, fit)
            candidates[mode] = fit
            print(f"  train NLL={fit['training_composite_nll']:.2f}; validation NLL="
                  f"{fit['validation_one_step_nll']:.2f}; converged={fit['optimizer_converged']}", flush=True)
        # Retain a spatial OU if learned transport cannot improve validation likelihood.
        selected = min(candidates, key=lambda mode: candidates[mode]["validation_one_step_nll"])
        fits.append(candidates[selected])
        all_fits[name] = {"candidates": candidates, "selected": selected}
    raw_correlation, coupling_method = filtered_driver_correlation(data, fits)
    # Only validation data choose the amount of coupling. alpha=0 is a serious
    # candidate, allowing the result to reject estimated cross-variable coupling.
    validation, best = {}, None
    for alpha in (0., .5, 1.):
        correlation = np.eye(4)+alpha*(raw_correlation-np.eye(4))
        score, _ = evaluate(data, system(data, fits, correlation), data.train_end,
                            data.validation_end, members=args.members, seed=args.seed)
        validation[str(alpha)] = score
        loss = score["joint_energy_score_per_sqrt_dimension"]
        if best is None or loss < best[0]:
            best = (loss, alpha, correlation)
        print(f"Validation coupling alpha={alpha}: joint energy={loss:.5f}", flush=True)
    _, alpha, correlation = best
    configurations = {
        "selected_joint_spde": (fits, correlation, "joint_spde"),
        "uncoupled_variables": (fits, np.eye(4), "joint_spde"),
        "independent_sites": (fits, correlation, "independent_sites"),
        "common_weather": (fits, correlation, "common_weather"),
        "spatial_ou": ([all_fits[name]["candidates"]["spatial_ou"] for name in FIELDS], correlation, "joint_spde"),
    }
    scores, extras = {}, {}
    for name, (parameters, corr, mode) in configurations.items():
        print(f"Final contiguous test: {name}", flush=True)
        matrices = system(data, parameters, corr, mode)
        score, extra = evaluate(data, matrices, data.validation_end, len(data.times), members=args.members,
                                seed=args.seed, store_paths=name == "selected_joint_spde")
        scores[name], extras[name] = score, extra
        if name == "selected_joint_spde":
            ensemble = sample_contiguous(data, matrices, extra["first_posterior"], data.validation_end,
                                         members=args.members, seed=args.seed+1)
            np.savez_compressed(out/"weather_ensemble_24h.npz", **ensemble)
            np.savez_compressed(out/"test_forecasts.npz", indices=extra["indices"], mean=extra["mean"],
                                variance=extra["variance"], lead_hours=extra["lead_hours"],
                                predictive_paths=extra["predictive_paths"])
        print(f"  joint energy={score['joint_energy_score_per_sqrt_dimension']:.5f}; "
              f"spatial variogram={score['mean_spatial_variogram']:.5f}", flush=True)
    report = {"status": "regional hourly reanalysis pilot; not feeder-scale weather validation",
              "source": data.metadata, "context": context, "seed": args.seed,
              "field_order": FIELDS, "grid_shape": [len(data.grid.y), len(data.grid.x)],
              "grid_x_m": data.grid.x.tolist(), "grid_y_m": data.grid.y.tolist(),
              "held_sites": data.sites.iloc[data.held_sites].site_id.tolist(),
              "training": {"start": str(data.times[0]), "end": str(data.times[data.train_end-1])},
              "validation": {"start": str(data.times[data.train_end]), "end": str(data.times[data.validation_end-1])},
              "test": {"start": str(data.times[data.validation_end]), "end": str(data.times[-1])},
              "background": {"formula": "affine coordinates + linear day trend + two daily harmonics; pooled training cells only",
                             "coefficients": data.coefficients.tolist(), "residual_scales": data.scale.tolist()},
              "marginal_models": all_fits, "selected_fits": fits,
              "coupling_fit": coupling_method, "raw_proxy_correlation": raw_correlation.tolist(),
              "selected_coupling_alpha": alpha, "driving_correlation": correlation.tolist(),
              "coupling_validation": validation, "test_scores": scores,
              "evaluation": {"rolling_horizon_hours": 6, "warmup_hours": 48,
                             "conditions": "32 nonheld cells, past data only; 4 central held cells never assimilated",
                             "interval_target": "ERA5 values, includes fitted independent model discrepancy",
                             "ensemble_24h": "separate single-origin process-only ensemble; not spliced rolling forecasts",
                             "baselines": "independent/common site ablations match average marginal stationary variance; not independently optimized"},
              "limitations": ["ERA5 is reanalysis, not independent station observations",
                              "one region and two months; no seasonal generalization claim",
                              "coarse grid cannot validate feeder-scale minute fluctuations",
                              "coupling is a filtered-proxy approximation, not exact joint maximum likelihood",
                              "this is a development pilot, not a preregistered benchmark; unfinished training optimization was resumed",
                              "event-free test periods cannot establish joint low-solar/low-wind tail calibration",
                              "constant coefficients and Gaussian residuals may miss changing weather regimes",
                              "24-hour ensemble is a demonstration; formal test scores cover 1..6-hour horizons"],
              "implementation": implementation_fingerprint()}
    write_json(out/"validation.json", report)
    report_sha256 = hashlib.sha256((out/"validation.json").read_bytes()).hexdigest()
    write_json(out/"model.json", regional_driver_model(report, report_sha256))
    data.sites.assign(held_out=np.isin(np.arange(len(data.sites)), data.held_sites)).to_csv(out/"sites.csv", index=False)
    plot_results(data, report, extras["selected_joint_spde"], out)
    print(f"Report: {out/'validation.json'}", flush=True)


def plot_results(data, report, extra, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    names = list(report["test_scores"])
    labels = ["Selected joint", "Uncoupled vars", "Independent sites", "Common weather", "Spatial OU"]
    axes[0, 0].bar(labels, [report["test_scores"][n]["joint_energy_score_per_sqrt_dimension"] for n in names], color="#39858c")
    axes[0, 0].set_title("Future test: joint energy score (lower better)")
    axes[0, 1].bar(labels, [report["test_scores"][n]["mean_spatial_variogram"] for n in names], color="#c1883c")
    axes[0, 1].set_title("Future test: mean spatial variogram (lower better)")
    for ax in axes[0]:
        ax.tick_params(axis="x", rotation=25)
    indices = extra["indices"][:48]
    site = data.held_sites[0]
    for ax, i in zip(axes[1], (0, 3)):
        mean = extra["mean"][:48, i, site]*data.scale[i]+data.background[indices, i, site]
        sd = np.sqrt(extra["variance"][:48, i, site])*data.scale[i]
        truth = data.raw[indices, i, site]
        if i == 0:
            from scipy.special import expit
            mean, lower, upper, truth = [1.6*expit(x) for x in (mean, mean-1.645*sd, mean+1.645*sd, truth)]
            daylight = data.clear[indices, site] >= 50
            for values in (mean, lower, upper, truth):
                values[~daylight] = np.nan
        else:
            lower, upper = mean-1.645*sd, mean+1.645*sd
        ax.plot(data.times[indices], truth, color="black", lw=1, label="ERA5 held cell")
        ax.plot(data.times[indices], mean, color="#237e8b", label="Forecast median")
        ax.fill_between(data.times[indices], lower, upper, color="#237e8b", alpha=.2, label="90% predictive interval")
        ax.set_title(f"Held cell {data.sites.site_id.iloc[site]}: {'clear-sky index' if i == 0 else 'temperature (C)'}")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    fig.suptitle("Regional hourly ERA5 pilot: frozen training fit, 6-hour rolling forecasts (UTC)")
    fig.tight_layout()
    fig.savefig(out/"validation.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit"))
    parser.add_argument("--prepared", type=Path, default=ROOT/"data/processed/spde/region")
    parser.add_argument("--output", type=Path, default=ROOT/"data/processed/spde/regional_driver")
    parser.add_argument("--start", default="2013-05-01")
    parser.add_argument("--end", default="2013-06-30")
    parser.add_argument("--latitudes", nargs="+", type=float, default=[50.25, 50.5, 50.75, 51., 51.25, 51.5])
    parser.add_argument("--longitudes", nargs="+", type=float, default=[5.75, 6., 6.25, 6.5, 6.75, 7.])
    parser.add_argument("--train-days", type=int, default=42)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--fit-budget", type=int, default=1000)
    parser.add_argument("--members", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.members < 2 or args.fit_budget < 20:
        parser.error("Use >=2 members and fit budget>=20.")
    if args.command == "prepare":
        path, url = fetch_region(ROOT/"data/raw/era5_regional", args.start, args.end,
                                 latitudes=args.latitudes, longitudes=args.longitudes)
        weather, sites, meta = prepare_region(path, url, args.start, args.end)
        args.prepared.mkdir(parents=True, exist_ok=True)
        weather.to_csv(args.prepared/"weather_hourly.csv.gz", index=False,
                       compression={"method": "gzip", "mtime": 0})
        sites.to_csv(args.prepared/"sites.csv", index=False)
        write_json(args.prepared/"provenance.json", meta)
        print(f"Prepared {len(sites)} distinct cells, {weather.timestamp_utc.nunique()} common hours")
    elif args.command == "fit":
        fit_experiment(args)


if __name__ == "__main__":
    main()
