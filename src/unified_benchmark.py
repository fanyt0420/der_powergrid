"""Common multiyear weather baselines, initialization and physical verification.

Training outcomes and held-cell truth never enter fitted baselines or input
libraries. Conditional forecasts use only 24 preceding hours. Calendar-only
scenarios initialize from season/hour-matched training histories, not test truth.
"""
from __future__ import annotations

import hashlib
import argparse
from copy import deepcopy
import inspect
import json
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.linalg import block_diag, solve_discrete_lyapunov
from scipy.special import expit
from scipy.stats import wasserstein_distance

from src.weather_spde import kalman_update
from src.regional_spde import SOLAR_CAP, covariance_factor, physical

PHYSICAL = ("ghi", "wind_speed_100m", "temp_air")
ROOT = Path(__file__).resolve().parents[1]
NAMES = {"neural_spde": "Neural SPDE", "fixed_spde": "Fixed SPDE", "kalman_fixed_spde": "Fixed SPDE + Kalman",
         "spatial_ar1": "Spatial AR(1)", "persistence": "Persistence + errors", "hourly_resample": "Hourly resampling",
         "history_blocks": "72h history blocks", "legacy_arma_copula": "Monthly ARMA/Copula"}
CONDITIONAL = ("neural_spde", "fixed_spde", "kalman_fixed_spde", "spatial_ar1", "persistence", "hourly_resample", "history_blocks")
SCENARIOS = ("neural_spde", "fixed_spde", "spatial_ar1", "hourly_resample", "history_blocks", "legacy_arma_copula")


# Shared scores, regional diagnostics and original ARMA/Copula adapter.
def geometry_weights(data):
    """Identity at available cells; inverse-distance interpolation at held cells."""
    xy, cells = data.grid.xy, data.train_sites
    weights = np.zeros((len(xy), len(cells)))
    weights[cells, np.arange(len(cells))] = 1.
    for cell in data.held_sites:
        distance = np.linalg.norm(xy[cells]-xy[cell], axis=1)
        nearest = np.argsort(distance)[:4]
        w = 1/np.maximum(distance[nearest], 1.)**2
        weights[cell, nearest] = w/w.sum()
    return weights


def training_library(data):
    """Residual histories reconstructed without ever reading held-cell outcomes.

    Solar is undefined at night. Missing training values are filled by the
    training hour-of-day mean, falling back to the training all-hour cell mean.
    These fills are only sampler inputs, never substituted for scoring targets.
    """
    z = np.take(data.z[:data.train_end], data.train_sites, axis=2).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fallback = np.nanmean(z, axis=0)
        for hour in range(24):
            select = data.times[:data.train_end].hour == hour
            clim = np.nanmean(z[select], axis=0)
            clim = np.where(np.isfinite(clim), clim, fallback)
            z[select] = np.where(np.isfinite(z[select]), z[select], clim)
    if not np.isfinite(z).all():
        raise ValueError("Insufficient finite training values for resampling.")
    return np.einsum("tfc,nc->tfn", z, geometry_weights(data))


def resample_paths(data, library, indices, members, rng, *, block_days):
    """Same UTC hour, training residuals only; whole-day blocks or independent hours."""
    indices = np.asarray(indices, int)
    if len(indices) == 0 or np.any(np.diff(indices) != 1):
        raise ValueError("Resampling targets must be consecutive.")
    training_hours = data.times[:data.train_end].hour.to_numpy()
    target_hours = data.times[indices].hour.to_numpy()
    result = np.empty((members, len(indices), 4, data.grid.size))
    chosen = None
    for j, hour in enumerate(target_hours):
        if not block_days or chosen is None or hour == 0:
            # Whole blocks cannot cross the end of training.
            remaining = min(24-hour, len(indices)-j) if block_days else 1
            candidates = np.flatnonzero((training_hours == hour)
                                        & (np.arange(len(library))+remaining <= len(library)))
            chosen = rng.choice(candidates, members)
        result[:, j] = library[chosen]
        if block_days:
            chosen = chosen+1
    return result


def _carry_forward(values, initial):
    last = np.array(initial, copy=True)
    result = np.empty_like(values)
    for i, row in enumerate(values):
        last = np.where(np.isfinite(row), row, last)
        result[i] = last
    return result


def persistence_forecast(data, library, origin, horizon, members, rng):
    """Latest observed residual + empirical training forecast-error trajectories.

    Uncertainty is calibrated from same-hour, multi-lead training errors. Thus
    persistence is not unfairly represented as a zero-spread point forecast.
    """
    cells = data.train_sites
    train = np.take(data.z[:data.train_end], cells, axis=2)
    initial = np.nanmean(train, axis=0)
    carried_train = _carry_forward(train, initial)
    available = np.take(data.z[:origin+1], cells, axis=2)
    latest = _carry_forward(available, initial)[-1]
    candidates = np.flatnonzero((data.times[:data.train_end].hour == data.times[origin].hour)
                                & (np.arange(data.train_end)+horizon < data.train_end))
    picked = rng.choice(candidates, members)
    observed_train = np.einsum("tfc,nc->tfn", carried_train, geometry_weights(data))
    anchor = np.einsum("fc,nc->fn", latest, geometry_weights(data))
    return np.stack([anchor+library[picked+lead]-observed_train[picked]
                     for lead in range(1, horizon+1)], axis=1)


def forecast_blocks(data, matrices, start, stop, members, seed, *, horizon=6):
    """Return origins and predictive paths. No future or held observations read."""
    f, q, initial, obs = matrices
    n, dim = data.grid.size, len(f)
    available = np.concatenate([i*n+data.train_sites for i in range(4)])
    h = np.eye(dim)[available]
    mean, covariance = np.zeros(dim), initial.copy()
    rng = np.random.default_rng(seed)
    factor_q = covariance_factor(q)
    for t in range(max(0, start-48), stop-1):
        mean = f@mean
        covariance = f@covariance@f.T+q
        mean, covariance, _ = kalman_update(mean, covariance, data.z[t].reshape(-1)[available],
                                           h, obs[available])
        if t+1 < start or (t+1-start) % horizon or t+1+horizon > stop:
            continue
        state = mean+rng.standard_normal((members, dim))@covariance_factor(covariance).T
        paths = []
        for _ in range(horizon):
            state = state@f.T+rng.standard_normal(state.shape)@factor_q.T
            observed = state+rng.standard_normal(state.shape)*obs
            paths.append(observed.reshape(members, 4, n))
        yield t, np.arange(t+1, t+1+horizon), np.stack(paths, axis=1)


def unconditional_paths(data, matrices, indices, members, seed):
    """Continuous stationary latent trajectory, no test-period assimilation.

    Observation discrepancy is added only to the verification view, not fed back
    into the latent dynamics. This matches the reanalysis target of other methods.
    """
    f, q, stationary, obs = matrices
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((members, len(f)))@covariance_factor(stationary).T
    factor = covariance_factor(q)
    paths = []
    for _ in indices:
        state = state@f.T+rng.standard_normal(state.shape)@factor.T
        paths.append((state+rng.standard_normal(state.shape)*obs).reshape(members, 4, data.grid.size))
    return np.stack(paths, axis=1)


def physical_scale(data):
    values = physical(data, data.z[:data.train_end], np.arange(data.train_end))
    return {name: float(np.nanstd(values[name][:, data.train_sites])) for name in PHYSICAL}


def empirical_crps(truth, samples):
    """Fair ensemble CRPS, with the off-diagonal U-statistic correction."""
    m = len(samples)
    if m < 2:
        raise ValueError("CRPS needs >=2 members.")
    ordered = np.sort(samples, axis=0)
    coefficient = (2*np.arange(1, m+1)-m-1).reshape((m,)+(1,)*(samples.ndim-1))
    return np.mean(abs(samples-truth), axis=0)-np.sum(coefficient*ordered, axis=0)/(m*(m-1))


def forecast_scores(data, indices, samples, scales, *, model):
    """Per-lead spatial and marginal scores; entire forecast blocks kept separate."""
    truth = physical(data, data.z[indices], indices)
    rows = []
    for group, cells in (("all", np.arange(data.grid.size)), ("held", data.held_sites)):
        standardized = np.stack([truth[k][:, cells]/scales[k] for k in PHYSICAL], axis=1)
        predicted = np.stack([samples[k][:, :, cells]/scales[k] for k in PHYSICAL], axis=2)
        ii, jj = np.triu_indices(len(cells), 1)
        distance = np.linalg.norm(data.grid.xy[cells[ii]]-data.grid.xy[cells[jj]], axis=1)
        weights = 1/np.maximum(distance, 500.)
        for t, index in enumerate(indices):
            valid = np.isfinite(standardized[t])
            x, y = predicted[:, t][:, valid], standardized[t][valid]
            energy = (np.linalg.norm(x-y, axis=1).mean()
                      -.5*np.linalg.norm(x-np.roll(x, 1, axis=0), axis=1).mean())/np.sqrt(len(y))
            variograms = []
            row = {"model": model, "group": group, "date": str(data.times[index].date()),
                   "timestamp_utc": str(data.times[index]), "lead_hour": t+1,
                   "energy_score": float(energy),
                   "crps_standardized": float(np.nanmean(empirical_crps(standardized[t], predicted[:, t])))}
            for k, name in enumerate(PHYSICAL):
                target, ensemble = truth[name][t, cells], samples[name][:, t, cells]
                mask = np.isfinite(target)
                lo, hi = np.quantile(ensemble, [.05, .95], axis=0)
                row[f"{name}_count"] = int(mask.sum())
                for metric, value in (
                    ("squared_error", (ensemble.mean(axis=0)-target)**2),
                    ("crps", empirical_crps(target, ensemble)),
                    ("coverage_90", (target >= lo) & (target <= hi)),
                    ("width_90", hi-lo)):
                    row[f"{name}_{metric}"] = float(np.mean(value[mask])) if mask.any() else np.nan
                pair = np.isfinite(standardized[t, k, ii]) & np.isfinite(standardized[t, k, jj])
                if pair.any():
                    delta = np.sqrt(abs(standardized[t, k, ii]-standardized[t, k, jj]))
                    expected = np.mean(np.sqrt(abs(predicted[:, t, k, ii]-predicted[:, t, k, jj])), axis=0)
                    variograms.append(float(np.average((delta[pair]-expected[pair])**2, weights=weights[pair])))
            row["spatial_variogram_score"] = float(np.mean(variograms))
            rows.append(row)
    return rows


def summarize_forecasts(rows):
    frame = pd.DataFrame(rows)
    records = []
    for (model, group), part in frame.groupby(["model", "group"], sort=False):
        record = {"model": model, "group": group, "scored_hours": len(part)}
        for metric in ("energy_score", "crps_standardized", "spatial_variogram_score"):
            record[metric] = float(part[metric].mean())
        for name in PHYSICAL:
            count = part[f"{name}_count"].to_numpy()
            for metric in ("squared_error", "crps", "coverage_90", "width_90"):
                values = part[f"{name}_{metric}"].to_numpy()
                good = (count > 0) & np.isfinite(values)
                value = float(np.average(values[good], weights=count[good]))
                key = f"{name}_{'rmse' if metric == 'squared_error' else metric}"
                record[key] = np.sqrt(value) if metric == "squared_error" else value
        records.append(record)
    return pd.DataFrame(records)


def paired_day_intervals(rows, *, reference="joint_spde", seed=91, draws=2000, block_days=1):
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    result = []
    for group in ("all", "held"):
        for metric in ("crps_standardized", "energy_score", "spatial_variogram_score"):
            daily = frame[frame.group == group].groupby(["date", "model"])[metric].mean().unstack()
            for baseline in daily.columns:
                if baseline == reference:
                    continue
                paired = daily[[reference, baseline]].dropna()
                difference = (paired[reference]-paired[baseline]).to_numpy()
                starts = rng.integers(0, len(difference), (draws, int(np.ceil(len(difference)/block_days))))
                picks = (starts[:, :, None]+np.arange(block_days)) % len(difference)
                picks = picks.reshape(draws, -1)[:, :len(difference)]
                resampled = difference[picks].mean(axis=1)
                base_score = float(paired[baseline].mean())
                result.append({"model": reference, "baseline": baseline, "group": group, "metric": metric,
                               "days": len(difference), "block_days": block_days,
                               "mean_difference": float(difference.mean()),
                               "ci_95_low": float(np.quantile(resampled, .025)),
                               "ci_95_high": float(np.quantile(resampled, .975)),
                               "skill": float(1-paired[reference].mean()/base_score) if base_score > 0 else None,
                               "interval_method": "paired circular moving-day-block bootstrap; short test limits inference"})
    return pd.DataFrame(result)


def lag_correlation(series, lag):
    """Compute within-trajectory, same-cell lag correlation; never join members."""
    a, b = series[:, :-lag], series[:, lag:]
    good = np.isfinite(a) & np.isfinite(b)
    count = good.sum(axis=1)
    ma = np.sum(np.where(good, a, 0.), axis=1)/np.maximum(count, 1)
    mb = np.sum(np.where(good, b, 0.), axis=1)/np.maximum(count, 1)
    da, db = np.where(good, a-ma[:, None], 0.), np.where(good, b-mb[:, None], 0.)
    denominator = np.sqrt(np.sum(da**2, axis=1)*np.sum(db**2, axis=1))
    return np.divide(np.sum(da*db, axis=1), denominator,
                     out=np.full_like(denominator, np.nan), where=(count >= 8) & (denominator > 1e-12))


def pair_correlations(series):
    m, _, n = series.shape
    result = np.full((m, n, n), np.nan)
    for i in range(n):
        for j in range(i+1, n):
            a, b = series[:, :, i], series[:, :, j]
            good = np.isfinite(a) & np.isfinite(b)
            count = good.sum(axis=1)
            ma = np.sum(np.where(good, a, 0.), axis=1)/np.maximum(count, 1)
            mb = np.sum(np.where(good, b, 0.), axis=1)/np.maximum(count, 1)
            da, db = np.where(good, a-ma[:, None], 0.), np.where(good, b-mb[:, None], 0.)
            denominator = np.sqrt(np.sum(da**2, axis=1)*np.sum(db**2, axis=1))
            correlation = np.divide(np.sum(da*db, axis=1), denominator,
                                    out=np.full(m, np.nan), where=(count >= 8) & (denominator > 1e-12))
            result[:, i, j] = result[:, j, i] = correlation
    return result


def run_lengths(mask):
    padded = np.r_[False, np.asarray(mask, bool), False].astype(int)
    return np.flatnonzero(np.diff(padded) == -1)-np.flatnonzero(np.diff(padded) == 1)


def _safe_mean(a):
    a = np.asarray(a)
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def generation_diagnostics(data, indices, samples, scales, *, model):
    """Observable physical quantities, with a training-only physical calendar trend removed.

    ACF/correlation use residual GHI-clear-sky-index, speed and temperature.
    Speed and temperature subtract the training calendar fit; Kc also subtracts
    its own physical calendar fit, avoiding logit assumptions for legacy weather.
    """
    from src.regional_spde import calendar_design
    cells = data.train_sites
    truth = physical(data, data.z[indices], indices)
    train = physical(data, data.z[:data.train_end], np.arange(data.train_end))
    train["ghi"] = SOLAR_CAP*expit(data.raw[:data.train_end, 0])
    residual_truth, residual_sample = {}, {}
    design = calendar_design(data.times, data.grid.xy, data.times[0])
    daylight = data.clear[indices] >= 50
    temporal, marginal, spatial, fleet = [], [], [], []
    correlations = {}
    for name in PHYSICAL:
        source = train[name]
        target = truth[name]
        ensemble = samples[name]
        if name == "ghi":
            target = np.divide(target, data.clear[indices], out=np.full_like(target, np.nan), where=daylight)
            ensemble = np.divide(ensemble, data.clear[indices][None], out=np.full_like(ensemble, np.nan), where=daylight[None])
        x = design[:data.train_end, cells].reshape(-1, 8)
        y = source[:, cells].reshape(-1)
        good = np.isfinite(y)
        coef = np.linalg.lstsq(x[good], y[good], rcond=None)[0]
        background = np.einsum("tnk,k->tn", design[indices], coef)
        residual_truth[name], residual_sample[name] = target-background, ensemble-background
        for lag in (1, 3, 6, 12, 24):
            if lag >= len(indices):
                continue
            real = lag_correlation(residual_truth[name][None], lag)[0]
            generated = lag_correlation(residual_sample[name], lag)
            temporal.append({"model": model, "variable": name, "lag_hours": lag,
                             "observed_acf": _safe_mean(real), "generated_acf": _safe_mean(generated),
                             "site_acf_mae": _safe_mean(abs(np.nanmean(generated, axis=0)-real))})
        real = pair_correlations(residual_truth[name][None])[0]
        member_correlations = pair_correlations(residual_sample[name])
        count = np.isfinite(member_correlations).sum(axis=0)
        generated = np.divide(np.nansum(member_correlations, axis=0), count,
                              out=np.full_like(real, np.nan), where=count > 0)
        correlations[name] = (real, generated)
        ii, jj = np.triu_indices(data.grid.size, 1)
        distances = np.linalg.norm(data.grid.xy[ii]-data.grid.xy[jj], axis=1)/1000
        for lower, upper in ((0, 30), (30, 60), (60, 100), (100, 200)):
            use = (distances >= lower) & (distances < upper) & np.isfinite(real[ii, jj]) & np.isfinite(generated[ii, jj])
            if use.any():
                spatial.append({"model": model, "variable": name, "distance_lower_km": lower,
                                "distance_upper_km": upper, "pairs": int(use.sum()),
                                "observed_correlation": float(real[ii[use], jj[use]].mean()),
                                "generated_correlation": float(generated[ii[use], jj[use]].mean()),
                                "correlation_mae": float(abs(real[ii[use], jj[use]]-generated[ii[use], jj[use]]).mean())})
        for group, chosen in (("all", np.arange(data.grid.size)), ("held", data.held_sites)):
            target_values = truth[name][:, chosen]
            generated_values = samples[name][:, :, chosen]
            mask = np.isfinite(target_values)
            real_flat = target_values[mask]
            sample_flat = generated_values[:, mask].ravel()
            marginal.append({"model": model, "group": group, "variable": name,
                             "observed_mean": float(real_flat.mean()), "generated_mean": float(sample_flat.mean()),
                             "observed_std": float(real_flat.std()), "generated_std": float(sample_flat.std()),
                             "wasserstein_standardized": float(wasserstein_distance(real_flat, sample_flat)/scales[name])})
        # Regional arithmetic mean weather, not actual electrical power.
        observed_region = np.mean(truth[name][:, cells], axis=-1)
        sampled_region = np.mean(samples[name][:, :, cells], axis=-1)
        ramp_mask = np.isfinite(observed_region[1:]) & np.isfinite(observed_region[:-1])
        real_ramp = abs(np.diff(observed_region))[ramp_mask]
        generated_ramp = abs(np.diff(sampled_region, axis=1))[:, ramp_mask]
        finite_region = np.isfinite(observed_region)
        fleet.append({"model": model, "variable": name,
                      "observed_regional_std": float(np.nanstd(observed_region)),
                      "generated_regional_std": float(np.nanstd(sampled_region[:, finite_region], axis=1).mean()),
                      "observed_abs_ramp_p95": float(np.quantile(real_ramp, .95)),
                      "generated_abs_ramp_p95": float(np.quantile(generated_ramp, .95))})
    # Training-only thresholds; night never counts as a low-solar event.
    train_kc = train["ghi"][:, cells]
    valid_train = np.isfinite(train_kc).all(axis=1)
    kc_threshold = float(np.quantile(train_kc[valid_train].mean(axis=1), .25))
    wind_threshold = float(np.quantile(train["wind_speed_100m"][valid_train][:, cells].mean(axis=1), .25))
    eligible = daylight[:, cells].all(axis=1)
    true_kc = SOLAR_CAP*expit(data.raw[indices, 0][:, cells]).mean(axis=-1)
    predicted_kc = np.divide(samples["ghi"][:, :, cells], data.clear[indices][:, cells][None],
                             out=np.zeros_like(samples["ghi"][:, :, cells]),
                             where=data.clear[indices][:, cells][None] >= 50).mean(axis=-1)
    real_wind = truth["wind_speed_100m"][:, cells].mean(axis=-1)
    sample_wind = samples["wind_speed_100m"][:, :, cells].mean(axis=-1)
    real_low = real_wind < wind_threshold
    sample_low = sample_wind < wind_threshold
    observed_event = (true_kc < kc_threshold) & real_low & eligible
    predicted_event = (predicted_kc < kc_threshold) & sample_low & eligible[None]
    probability = predicted_event[:, eligible].mean(axis=0)
    lengths_true = run_lengths(observed_event)
    lengths_sample = np.concatenate([run_lengths(event) for event in predicted_event])
    wind_true = run_lengths(real_low)
    wind_sample = np.concatenate([run_lengths(event) for event in sample_low])
    cross_variable = []
    for a, b in (("ghi", "wind_speed_100m"), ("ghi", "temp_air"), ("wind_speed_100m", "temp_air")):
        real = np.stack([residual_truth[a], residual_truth[b]], axis=-1).reshape(1, -1, 2)
        generated = np.stack([residual_sample[a], residual_sample[b]], axis=-1).reshape(len(samples[a]), -1, 2)
        observed_corr = pair_correlations(real)[0, 0, 1]
        generated_corr = _safe_mean(pair_correlations(generated)[:, 0, 1])
        cross_variable.append({"model": model, "variable_pair": a+"/"+b,
                               "observed_pooled_residual_correlation": float(observed_corr),
                               "generated_pooled_residual_correlation": generated_corr,
                               "absolute_error": float(abs(generated_corr-observed_corr))})
    event = {"model": model, "solar_kc_threshold": kc_threshold, "wind_threshold_m_s": wind_threshold,
             "daylight_hours": int(eligible.sum()), "observed_events": int(observed_event.sum()),
             "observed_compound_frequency": float(observed_event[eligible].mean()),
             "generated_compound_frequency": float(probability.mean()),
             "brier_score": float(np.mean((probability-observed_event[eligible])**2)),
             "observed_compound_max_run_hours": int(lengths_true.max()) if len(lengths_true) else 0,
             "generated_compound_run_p95_hours": float(np.quantile(lengths_sample, .95)) if len(lengths_sample) else 0.,
             "observed_low_wind_max_run_hours": int(wind_true.max()) if len(wind_true) else 0,
             "generated_low_wind_member_max_run_median_hours": float(np.median([max(run_lengths(x), default=0) for x in sample_low])),
             "generated_low_wind_run_p95_hours": float(np.quantile(wind_sample, .95)) if len(wind_sample) else 0.,
             "event_calibration_identifiable": bool(observed_event.any() and (~observed_event[eligible]).any())}
    # A plot of one representative available cell; no best member selection.
    cell = int(cells[len(cells)//2])
    band = {name: np.quantile(samples[name][:, :, cell], [.05, .5, .95], axis=0) for name in PHYSICAL}
    return {"temporal": temporal, "marginal": marginal, "spatial": spatial, "cross_variable": cross_variable,
            "regional_weather": fleet, "events": event}, band


def trajectory_scores(data, indices, samples, scales, *, model):
    """Whole-trajectory energy and temporal variogram scores, one test block.

    This is a probabilistic verification view of the entire continuous path,
    distinct from diagnostic ACF matching. No transition across independent
    origins or members is introduced.
    """
    truth = physical(data, data.z[indices], indices)
    y = np.stack([truth[k]/scales[k] for k in PHYSICAL], axis=1)
    x = np.stack([samples[k]/scales[k] for k in PHYSICAL], axis=2)
    good = np.isfinite(y)
    flat_x, flat_y = x[:, good], y[good]
    energy = (np.linalg.norm(flat_x-flat_y, axis=1).mean()
              -.5*np.linalg.norm(flat_x-np.roll(flat_x, 1, axis=0), axis=1).mean())/np.sqrt(len(flat_y))
    temporal_scores, weights = [], []
    for lag in (1, 3, 6, 12, 24):
        if lag >= len(indices):
            continue
        delta = np.sqrt(abs(y[lag:]-y[:-lag]))
        expected = np.mean(np.sqrt(abs(x[:, lag:]-x[:, :-lag])), axis=0)
        temporal_scores.append(float(np.nanmean((delta-expected)**2)))
        weights.append(1/lag)
    return {"model": model, "scored_hours": len(indices),
            "trajectory_energy_score": float(energy),
            "temporal_variogram_score": float(np.average(temporal_scores, weights=weights))}


def fit_legacy_group(data):
    """Short-record adapter using the existing generator's fitting primitives.

    One nonheld central source cell; all 42 training days pooled into one group
    because a separately fitted June group has <20 observations per hour. The
    original VAR routine's <50-day fallback is retained and disclosed.
    """
    from data.generate_data import era5_synth as legacy
    from scipy.stats import kurtosis
    cell = int(data.train_sites[np.argmin(np.linalg.norm(data.grid.xy[data.train_sites], axis=1))])
    train = physical(data, data.z[:data.train_end], np.arange(data.train_end))
    values = {"kt": np.minimum(SOLAR_CAP*expit(data.raw[:data.train_end, 0, cell]), legacy.KT_MAX),
              "temp_air": train["temp_air"][:, cell],
              "wind_speed_100m": train["wind_speed_100m"][:, cell], "wind_speed": data.wind10[:data.train_end, cell]}
    hours = data.times[:data.train_end].hour.to_numpy()
    days = data.times[:data.train_end].normalize().asi8
    unique, inv = np.unique(days, return_inverse=True)
    clim, anom, scores = {}, {}, {}
    for name in legacy.VARS:
        mean, sd = np.full(24, np.nan), np.full(24, np.nan)
        for h in range(24):
            x = values[name][hours == h]
            x = x[np.isfinite(x)]
            if len(x) >= 20:
                mean[h], sd[h] = x.mean(), x.std(ddof=1)
        sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, np.nan)
        anom[name] = (values[name]-mean[hours])/sd[hours]
        scores[name] = legacy.normal_scores(anom[name])
        clim[name] = {"mean": np.nan_to_num(mean).tolist(), "std": np.nan_to_num(sd, nan=1.).tolist()}
    daily = np.stack([np.array([np.nanmean(scores[name][inv == d]) for d in range(len(unique))])
                      for name in legacy.VARS], axis=1)
    consecutive = np.flatnonzero(np.r_[False, np.diff(unique) == 86400*10**9])
    var1 = legacy._fit_var1(daily, consecutive)
    arma, residual_sd = {}, {}
    same_day = np.r_[False, inv[1:] == inv[:-1]]
    for k, name in enumerate(legacy.VARS):
        residual = scores[name]-daily[inv, k]
        sd = max(float(np.nanstd(residual)), 1e-9)
        picked = legacy.fit_arima(residual/sd)
        increments = legacy._within_day_diff(scores[name], same_day)
        gamma = float(kurtosis(increments)) if len(increments) > 200 else np.nan
        picked["nu"] = legacy._kurtosis_to_nu(legacy._innovation_kurtosis(gamma, picked["ar"], picked["ma"]))
        arma[name], residual_sd[name] = picked, sd
    complete = np.column_stack([scores[name] for name in legacy.VARS])
    complete = complete[np.isfinite(complete).all(axis=1)]
    corr = np.corrcoef(complete.T)
    innovation = legacy._innovation_corr(corr, {k: (v["ar"], v["ma"]) for k, v in arma.items()})
    nus = [v["nu"] for v in arma.values() if v["nu"] is not None]
    group = {"clim": clim, "quantiles": {k: legacy.quantile_grid(anom[k]) for k in legacy.VARS},
             "var1": var1, "resid_sd": residual_sd, "arma": arma,
             "innov_nu": float(np.median(nus)) if nus else None, "innov_corr": innovation.tolist()}
    return {"source_cell": cell, "source_site": str(data.sites.site_id.iloc[cell]),
            "training_days": len(unique), "daily_var_fallback": var1["n_pairs"] < 50,
            "months": {str(m): deepcopy(group) for m in set(data.times.month)},
            "limitations": ["Short-record pooled group, not the original multiyear monthly model",
                            "One nonheld source; generated Kc, speed and T shared across cells; local clear sky retained",
                            "Original generator's Gaussian-CDF mapping after t noise retained",
                            "Original daily ARMA filter restart and realized monthly scaling retained",
                            "Original daily VAR <50-pair fallback retained; daily persistence not fitted"]}


def legacy_paths(data, fitted, indices, members, seed):
    """Reuse the unchanged legacy sampler with an ERA5-aligned clear-sky adapter."""
    from data.generate_data import era5_synth as legacy
    from unittest.mock import patch
    if len(indices) % 24 or data.times[indices[0]].hour != 0:
        raise ValueError("Legacy comparison requires complete UTC days.")
    cell = fitted["source_cell"]
    def clear_sky_adapter(site, naive_index):
        midpoint = pd.DatetimeIndex(naive_index).tz_localize("UTC")+pd.Timedelta(minutes=30)
        positions = data.times.get_indexer(midpoint)
        if (positions < 0).any():
            raise ValueError("Legacy dates outside aligned weather.")
        clear = data.clear[positions, cell]
        # Zenith affects only discarded DNI/DHI in the legacy sampler.
        return pd.DataFrame({"ghi_clear": clear, "ghi_clear_avg": clear, "zenith": 45.}, index=naive_index)
    dates = list(data.times[indices][::24].tz_localize(None).normalize())
    output = {k: [] for k in PHYSICAL}
    with patch.object(legacy, "_clearsky", clear_sky_adapter):
        for member in range(members):
            frame = legacy.sample_days("benchmark_adapter", fitted, dates, seed=seed+member)
            clear = data.clear[indices, cell]
            kc = np.divide(frame.ghi.to_numpy(), clear, out=np.zeros(len(indices)), where=clear > 0)
            output["ghi"].append(kc[:, None]*data.clear[indices])
            for name in PHYSICAL[1:]:
                output[name].append(np.broadcast_to(frame[name].to_numpy()[:, None], (len(indices), data.grid.size)).copy())
    result = {k: np.stack(v) for k, v in output.items()}
    # Match the same daytime verification support as all other methods.
    result["ghi"][:, data.clear[indices] < 50] = np.nan
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive_factor(covariance):
    values, vectors = np.linalg.eigh((covariance+covariance.T)/2)
    tolerance = 1e-6*max(1., float(values.max()))
    if values.min() < -tolerance:
        raise ValueError("Covariance is not positive semidefinite.")
    return vectors*np.sqrt(np.maximum(values, 0.))


def candidates(data, month, hour, horizon=72, history=24):
    available = data.origins("train", history=history, horizon=horizon)
    starts = data.times[available+1]
    chosen = available[(starts.month == month) & (starts.hour == hour)]
    if not len(chosen):
        raise ValueError("No training blocks match the requested season/hour/horizon.")
    return chosen


def prior_histories(data, origin, members, rng, horizon=72):
    target = data.times[origin+1]
    selected = rng.choice(candidates(data, target.month, target.hour, horizon), members)
    return data.input_z[selected[:, None]+np.arange(-23, 1)], selected


def history_anchor(data, origins):
    """Latest visible value within exactly 24h; no held-cell observations."""
    origins = np.asarray(origins, int)
    cells = data.train_sites
    last = np.zeros((len(origins), 4, len(cells)))
    for offset in range(-23, 1):
        values = data.z[origins+offset][:, :, cells]
        valid = data.valid[origins+offset][:, :, cells]
        last = np.where(valid, values, last)
    return np.einsum("bfc,nc->bfn", last, data.projection[:, cells])


def resample(data, origin, horizon, members, rng, *, blocks):
    target = data.times[origin+1]
    if blocks:
        selected = rng.choice(candidates(data, target.month, target.hour, horizon), members)
        return data.input_z[selected[:, None]+np.arange(1, horizon+1)]
    result = []
    training = data.times.year.isin(data.manifest["split"]["train"])
    for index in range(origin+1, origin+horizon+1):
        time = data.times[index]
        available = np.flatnonzero(training & (data.times.month == time.month) & (data.times.hour == time.hour))
        result.append(data.input_z[rng.choice(available, members)])
    return np.stack(result, axis=1)


def persistence(data, origin, horizon, members, rng):
    target = data.times[origin+1]
    selected = rng.choice(candidates(data, target.month, target.hour, horizon), members)
    anchor = history_anchor(data, [origin])[0]
    errors = data.input_z[selected[:, None]+np.arange(1, horizon+1)]-history_anchor(data, selected)[:, None]
    return anchor[None, None]+errors


def fit_spatial_ar1(data):
    """Pooled per-field AR(1), empirical spatial/cross-field innovation covariance.

    All transitions from training years and visible cells are used. Pairwise
    covariance handles missing solar, with fixed 5% diagonal shrinkage and an
    eigenvalue floor. Held innovations are interpolated, never estimated there.
    """
    train = data.times.year.isin(data.manifest["split"]["train"])
    t = np.flatnonzero(train[1:] & train[:-1])+1
    cells = data.train_sites
    x, y = data.z[t-1][:, :, cells].astype(float), data.z[t][:, :, cells].astype(float)
    valid = data.valid[t-1][:, :, cells] & data.valid[t][:, :, cells]
    rho, drift = [], []
    for f in range(4):
        a, b = x[:, f][valid[:, f]], y[:, f][valid[:, f]]
        if len(a) < 100:
            raise ValueError("Too few training transitions for spatial AR1.")
        phi = float(np.clip(np.mean((a-a.mean())*(b-b.mean()))/max(a.var(), 1e-9), .01, .995))
        rho.append(phi)
        drift.append(float(b.mean()-phi*a.mean()))
    residual = y-np.array(rho)[None, :, None]*x-np.array(drift)[None, :, None]
    values = np.where(valid, residual, 0.).reshape(len(t), -1)
    mask = valid.reshape(len(t), -1).astype(float)
    count = mask.T@mask
    mean_pair = (values.T@mask)/np.maximum(count, 1.)
    covariance = (values.T@values)/np.maximum(count, 1.)-mean_pair*mean_pair.T
    covariance = .95*covariance+.05*np.diag(np.diag(covariance))
    w, v = np.linalg.eigh((covariance+covariance.T)/2)
    floor = max(float(w.max())*1e-6, 1e-8)
    covariance = (v*np.maximum(w, floor))@v.T
    h = block_diag(*([data.projection[:, cells]]*4))
    q = h@covariance@h.T
    diagonal = np.repeat(rho, data.grid.size)
    f = np.diag(diagonal)
    p = q/(1-diagonal[:, None]*diagonal[None])
    return {"f": f, "q": q, "initial": p, "drift": np.repeat(drift, data.grid.size),
            "observation_std": .025, "fit": {"rho": rho, "drift": drift,
                "training_transitions": len(t), "visible_cells": len(cells),
                "diagonal_shrinkage": .05, "covariance_eigenvalue_floor": floor}}


def fixed_linear(model):
    """Exact linear state system of the retained constant-coefficient model."""
    import torch
    if model.neural:
        raise ValueError("Only the constant SPDE admits this fixed linear system.")
    device = next(model.parameters()).device
    with torch.no_grad():
        history = torch.zeros(1, model.history, 4, model.size, device=device)
        calendar = torch.zeros(1, 11, device=device)
        parameters = [p.cpu().numpy()[0] for p in model.parameters_at(history, calendar)]
        mean, g, factor = model.matrices(history, calendar)
    vx, vy, diffusion, decay = parameters[:4]
    matrices = []
    for k in range(4):
        split = lambda value: ((np.sqrt(value*value+1)+value)/2, (np.sqrt(value*value+1)-value)/2)
        xp, xn = split(vx[k]); yp, yn = split(vy[k])
        operator = (diffusion[k]*model.lap.cpu().numpy()+xp*model.axp.cpu().numpy()+xn*model.axn.cpu().numpy()
                    +yp*model.ayp.cpu().numpy()+yn*model.ayn.cpu().numpy()-decay[k]*np.eye(model.size))
        matrices.append(np.linalg.solve(np.eye(model.size)-operator, np.eye(model.size)))
    f = block_diag(*matrices)
    drive = block_diag(*g.cpu().numpy()[0])@np.kron(factor.cpu().numpy()[0], np.eye(model.size))
    q = drive@drive.T
    p = solve_discrete_lyapunov(f, q)
    return {"f": f, "q": q, "initial": (p+p.T)/2, "drift": mean.cpu().numpy().reshape(-1),
            "observation_std": .025}


def linear_paths(data, system, origin, horizon, members, rng, *, histories=None):
    f, q, p, drift = (system[key] for key in ("f", "q", "initial", "drift"))
    n = data.grid.size
    if histories is None:
        # Exact 24h common information window, no test-future observations.
        mean = np.linalg.solve(np.eye(len(f))-f, drift)
        covariance = p.copy()
        visible = np.concatenate([field*n+data.train_sites for field in range(4)])
        h = np.eye(len(f))[visible]
        for index in range(origin-23, origin+1):
            mean = f@mean+drift
            covariance = f@covariance@f.T+q
            row = data.z[index].reshape(-1)[visible].copy()
            row[~data.valid[index].reshape(-1)[visible]] = np.nan
            mean, covariance, _ = kalman_update(mean, covariance, row, h, system["observation_std"])
        state = mean+rng.standard_normal((members, len(f)))@positive_factor(covariance).T
    else:
        state = histories[:, -1].copy().reshape(members, -1)
        hidden = np.all(histories[:, -1, 0] == 0, axis=1)
        # Nighttime solar is unknown; use this model's solar prior, not test truth.
        prior = rng.standard_normal((members, n))@positive_factor(p[:n, :n]).T
        state[:, :n] = np.where(hidden[:, None], prior, state[:, :n])
    factor = positive_factor(q)
    result = []
    for _ in range(horizon):
        state = state@f.T+drift+rng.standard_normal(state.shape)@factor.T
        result.append(state.reshape(members, 4, n))
    return np.stack(result, axis=1)


def fit_monthly_legacy(data, output, project_root):
    """Refit unchanged original monthly ARMA/Copula primitives on one visible cell.

    True 10m wind is read from the same training-quarter cache, not inferred from
    100m wind. The only adapters replace data loading/clear-sky source and output
    location. Original fitting and sampling quirks are preserved and disclosed.
    """
    from data.generate_data import era5_synth as legacy
    root, output = Path(project_root), Path(output)
    cell = int(data.train_sites[np.argmin(np.linalg.norm(data.grid.xy[data.train_sites], axis=1))])
    wind10 = np.full(len(data.times), np.nan)
    provenance = []
    station = data.sites.iloc[cell]
    for year in data.manifest["split"]["train"]:
        for quarter in data.manifest["years"][str(year)]["quarters"]:
            cached = root/"data/raw/era5_neural"/Path(quarter["source_file"]).name
            if sha(cached) != quarter["source_sha256"]:
                raise ValueError("Legacy training-quarter checksum mismatch.")
            payload = json.loads(cached.read_text(encoding="utf-8-sig"))
            points = [p for p in payload if np.isclose(p["latitude"], station.latitude)
                      and np.isclose(p["longitude"], station.longitude)]
            if len(points) != 1 or points[0]["hourly_units"]["wind_speed_10m"] != "m/s":
                raise ValueError("Legacy source cell or 10m wind unit differs.")
            hourly = points[0]["hourly"]
            endpoint = pd.to_datetime(hourly["time"], utc=True)
            middle = endpoint[1:]-pd.Timedelta(minutes=30)
            values = np.asarray(hourly["wind_speed_10m"], float)
            values = .5*(values[1:]+values[:-1])
            positions = data.times.get_indexer(middle)
            valid = (positions >= 0) & (middle >= pd.Timestamp(quarter["start"])) & (middle < pd.Timestamp(quarter["end_exclusive"]))
            wind10[positions[valid]] = values[valid]
            provenance.append({"file": str(cached.relative_to(root)), "sha256": quarter["source_sha256"]})
            del payload
    use = data.times.year.isin(data.manifest["split"]["train"])
    if not np.isfinite(wind10[use]).all():
        raise ValueError("Missing legacy training 10m winds.")
    raw = data.raw[use, :, cell]
    clear = data.clear[use, cell]
    kt = np.divide(data.ghi[use, cell], clear, out=np.full_like(clear, np.nan), where=clear >= legacy.KT_MIN_CLEAR)
    frame = pd.DataFrame({"kt": np.clip(kt, 0, legacy.KT_MAX),
        "temp_air": raw[:, 3], "wind_speed_100m": np.hypot(raw[:, 1], raw[:, 2]),
        "wind_speed": wind10[use]}, index=data.times[use].tz_localize(None))
    reference = "regional_reference"
    with patch.object(legacy, "load_years", lambda *a, **k: frame), \
         patch.object(legacy, "clearness", lambda raw, site: raw), \
         patch.object(legacy, "SYNTH_DIR", output), \
         patch.dict(legacy.SITES, {reference: {"label": "ERA5 visible regional reference"}}):
        model = legacy.fit(reference, (min(data.manifest["split"]["train"]), max(data.manifest["split"]["train"])))
    if len(model["months"]) != 12:
        raise ValueError("Legacy monthly refit did not cover all 12 months.")
    model.update(source_cell=cell, source_site=str(station.site_id), training_quarters=provenance)
    return model


def score_weather(data, indices, samples, scales, model, protocol, origin_label):
    """Identical daylight/finite-target support for every model; per-origin rows."""
    truth = {"ghi": data.ghi[indices], "wind_speed_100m": np.hypot(data.raw[indices, 1], data.raw[indices, 2]),
             "temp_air": data.raw[indices, 3]}
    rows, joint, structure = [], [], []
    for group, cells in (("all", np.arange(data.grid.size)), ("held", data.held)):
        for name in PHYSICAL:
            y, x = truth[name][:, cells], samples[name][:, :, cells]
            valid = np.isfinite(y)
            if name == "ghi":
                valid &= data.clear[indices][:, cells] >= 50
            if not np.isfinite(x[:, valid]).all():
                raise ValueError(f"Nonfinite generated targets: {model}/{protocol}/{name}.")
            lo, hi = np.quantile(x, [.05, .95], axis=0)
            crps = empirical_crps(y, x)
            interval = hi-lo+20*np.maximum(lo-y, 0)+20*np.maximum(y-hi, 0)
            for label, begin, end in (("all", 0, len(indices)), ("1-6h", 0, 6), ("7-24h", 6, 24), ("25-72h", 24, len(indices))):
                mask = valid[begin:end]
                if not mask.any():
                    continue
                rows.append({"model": model, "protocol": protocol, "origin": origin_label, "group": group,
                    "variable": name, "lead_band": label, "target_count": int(mask.sum()),
                    "crps_standardized": float(crps[begin:end][mask].mean()/scales[name]),
                    "rmse": float(np.sqrt(((x.mean(axis=0)-y)**2)[begin:end][mask].mean())),
                    "coverage_90": float(((y >= lo) & (y <= hi))[begin:end][mask].mean()),
                    "width_90": float((hi-lo)[begin:end][mask].mean()),
                    "interval_score_90_standardized": float(interval[begin:end][mask].mean()/scales[name]),
                    "observed_mean": float(y[begin:end][mask].mean()),
                    "generated_mean": float(x[:, begin:end][:, mask].mean())})
        y = np.stack([truth[k][:, cells]/scales[k] for k in PHYSICAL], axis=1)
        x = np.stack([samples[k][:, :, cells]/scales[k] for k in PHYSICAL], axis=2)
        valid = np.isfinite(y)
        valid[:, 0] &= data.clear[indices][:, cells] >= 50
        flat_x, flat_y = x[:, valid], y[valid]
        energy = (np.linalg.norm(flat_x-flat_y, axis=1).mean()
                  -.5*np.linalg.norm(flat_x-np.roll(flat_x, 1, axis=0), axis=1).mean())/np.sqrt(len(flat_y))
        temporal = []
        for lag in (1, 6, 24):
            observed = np.sqrt(abs(y[lag:]-y[:-lag]))
            expected = np.sqrt(abs(x[:, lag:]-x[:, :-lag])).mean(axis=0)
            temporal.append(float(((observed-expected)**2)[valid[lag:] & valid[:-lag]].mean()))
        i, j = np.triu_indices(len(cells), 1)
        observed = np.sqrt(abs(y[..., i]-y[..., j]))
        expected = np.sqrt(abs(x[..., i]-x[..., j])).mean(axis=0)
        pair_mask = valid[..., i] & valid[..., j]
        spatial = float(((observed-expected)**2)[pair_mask].mean()) if pair_mask.any() else np.nan
        joint.append({"model": model, "protocol": protocol, "origin": origin_label, "group": group,
                      "trajectory_energy": float(energy), "temporal_variogram": float(np.mean(temporal)),
                      "spatial_variogram": spatial})
    # Remove known physical seasonal background before correlation diagnostics.
    background = {"ghi": 1.6*expit(data.background[indices, 0]),
        "wind_speed_100m": np.hypot(data.background[indices, 1], data.background[indices, 2]),
        "temp_air": data.background[indices, 3]}
    reference, generated = [], []
    for name in PHYSICAL:
        y, x = truth[name].copy(), samples[name].copy()
        if name == "ghi":
            support = data.clear[indices] >= 50
            y = np.divide(y, data.clear[indices], out=np.full_like(y, np.nan), where=support)
            x = np.divide(x, data.clear[indices][None], out=np.full_like(x, np.nan), where=support[None])
        y, x = y-background[name], x-background[name][None]
        reference.append(y); generated.append(x)
        for lag in (1, 6, 24):
            observed, members = lag_correlation(y[None], lag)[0], lag_correlation(x, lag)
            count = np.isfinite(members).sum(axis=0)
            estimated = np.divide(np.nansum(members, axis=0), count, out=np.full_like(observed, np.nan), where=count > 0)
            good = np.isfinite(observed) & np.isfinite(estimated)
            if good.any():
                structure.append({"model": model, "protocol": protocol, "origin": origin_label, "variable": name,
                    "metric": "acf", "lag_hours": lag, "observed": float(observed[good].mean()),
                    "generated": float(estimated[good].mean()), "mae": float(abs(observed[good]-estimated[good]).mean())})
        observed = pair_correlations(y[None])[0]
        values = pair_correlations(x)
        count = np.isfinite(values).sum(axis=0)
        estimated = np.divide(np.nansum(values, axis=0), count, out=np.full_like(observed, np.nan), where=count > 0)
        i, j = np.triu_indices(data.grid.size, 1)
        good = np.isfinite(observed[i, j]) & np.isfinite(estimated[i, j])
        if good.any():
            structure.append({"model": model, "protocol": protocol, "origin": origin_label, "variable": name,
                "metric": "spatial_correlation", "lag_hours": 0, "observed": float(observed[i[good], j[good]].mean()),
                "generated": float(estimated[i[good], j[good]].mean()),
                "mae": float(abs(observed[i[good], j[good]]-estimated[i[good], j[good]]).mean())})
    for first, second in ((0, 1), (0, 2), (1, 2)):
        observed_pair = np.stack((reference[first], reference[second]), axis=-1).reshape(-1, 2)
        simulated_pair = np.stack((generated[first], generated[second]), axis=-1).reshape(len(samples["ghi"]), -1, 2)
        real_corr = pair_correlations(observed_pair[None])[0, 0, 1]
        sim_corr = pair_correlations(simulated_pair)[:, 0, 1]
        if np.isfinite(real_corr) and np.isfinite(sim_corr).any():
            structure.append({"model": model, "protocol": protocol, "origin": origin_label,
                "variable": PHYSICAL[first]+"/"+PHYSICAL[second], "metric": "cross_variable_correlation", "lag_hours": 0,
                "observed": float(real_corr), "generated": float(np.nanmean(sim_corr)),
                "mae": float(np.nanmean(abs(sim_corr-real_corr)))})
    return rows, joint, structure


def regional_series(clear, ghi, wind, cells):
    """Daylight mean Kc and all-hour mean wind; same mask for every method."""
    daylight = clear[:, cells] >= 50
    kc = np.divide(ghi[..., cells], clear[:, cells], out=np.zeros_like(ghi[..., cells]), where=daylight)
    count = daylight.sum(axis=-1)
    average = kc.sum(axis=-1)/np.maximum(count, 1)
    return average, wind[..., cells].mean(axis=-1), count >= max(1, len(cells)/2)


def event_thresholds(data):
    train = np.flatnonzero(data.times.year.isin(data.manifest["split"]["train"]))
    wind = np.hypot(data.raw[train, 1], data.raw[train, 2])
    kc, speed, daylight = regional_series(data.clear[train], data.ghi[train], wind, data.train_sites)
    return {"low_kc": float(np.quantile(kc[daylight], .25)), "low_wind": float(np.quantile(speed[daylight], .25))}


def event_score(data, indices, samples, thresholds, model, protocol, origin):
    real_wind = np.hypot(data.raw[indices, 1], data.raw[indices, 2])
    yk, yw, day = regional_series(data.clear[indices], data.ghi[indices], real_wind, data.train_sites)
    xk, xw, _ = regional_series(data.clear[indices], samples["ghi"], samples["wind_speed_100m"], data.train_sites)
    real = (yk < thresholds["low_kc"]) & (yw < thresholds["low_wind"])
    generated = (xk < thresholds["low_kc"]) & (xw < thresholds["low_wind"])
    probability = generated.mean(axis=0)
    longest = lambda a: int(run_lengths(a).max()) if np.any(a) else 0
    return {"model": model, "protocol": protocol, "origin": origin,
        "daylight_hours": int(day.sum()), "observed_events": int(real[day].sum()),
        "brier": float(((probability-real)**2)[day].mean()),
        "observed_frequency": float(real[day].mean()), "generated_frequency": float(generated[:, day].mean()),
        "observed_longest_compound_run": longest(real & day),
        "generated_longest_compound_run_median": float(np.median([longest(a & day) for a in generated])),
        "window_censoring": "Runs never cross origin/member boundaries; edges can be censored."}


def paired_intervals(rows, reference="neural_spde", *, seed=20261025, draws=5000):
    frame = pd.DataFrame(rows)
    frame = frame[frame.lead_band == "all"]
    scores = frame.groupby(["split", "protocol", "group", "origin", "model"]).crps_standardized.mean()
    records, rng = [], np.random.default_rng(seed)
    for (split, protocol, group), part in scores.groupby(level=["split", "protocol", "group"]):
        table = part.droplevel(["split", "protocol", "group"]).unstack("model")
        chosen = rng.integers(0, len(table), (draws, len(table)))
        for baseline in table.columns:
            if baseline == reference:
                continue
            delta = table[reference].to_numpy()-table[baseline].to_numpy()
            lo, hi = np.quantile(delta[chosen].mean(axis=1), [.025, .975])
            records.append({"split": split, "protocol": protocol, "group": group, "reference": reference,
                "baseline": baseline, "origins": len(table), "mean_crps_difference": float(delta.mean()),
                "ci_95_low": float(lo), "ci_95_high": float(hi),
                "skill": float(1-table[reference].mean()/table[baseline].mean())})
    return pd.DataFrame(records)


def load_models(data, directory, device):
    from src.neural_weather_spde import CoefficientSPDE
    saved = json.loads((directory/"protocol.json").read_text())
    if saved["context"]["data_manifest_sha256"] != sha(data.directory/"manifest.json") or saved["held_sites"] != data.held.tolist():
        raise ValueError("Models and benchmark must use the same data manifest and spatial holdout.")
    with np.load(directory/"preprocessing.npz") as original:
        for key in ("coefficients", "scale", "projection"):
            np.testing.assert_array_equal(original[key], getattr(data, key))
    config = saved["context"]
    if config["history"] != 24:
        raise ValueError("This protocol fixes a common 24h information window.")
    models = {}
    for name, file in (("neural_spde", "neural_best.pt"), ("fixed_spde", "constant_best.pt")):
        model = CoefficientSPDE(data.grid, data.projection, history=24, width=config["width"],
                                depth=config["depth"], neural=name == "neural_spde").to(device)
        model.load_state_dict(torch.load(directory/file, map_location=device, weights_only=True))
        models[name] = model.eval()
    return models


@torch.no_grad()
def neural_paths(model, data, origin, members, device, seed, histories=None):
    torch.manual_seed(seed)
    indices = np.arange(origin+1, origin+73)
    calendar = torch.as_tensor(data.calendar[indices][None], dtype=torch.float32, device=device)
    daylight = torch.as_tensor((data.clear[indices] >= 50)[None], device=device)
    if histories is None:
        history = torch.as_tensor(data.input_z[origin-23:origin+1][None], dtype=torch.float32, device=device)
        return model.rollout(history, calendar, daylight, members=members)[0].cpu().numpy()
    history = torch.as_tensor(histories, dtype=torch.float32, device=device)
    paths = model.rollout(history, calendar.expand(members, -1, -1),
                          daylight.expand(members, -1, -1), members=1)
    return paths[:, 0].cpu().numpy()


def main():
    from src.neural_weather_spde import MultiYearWeather
    from data.generate_data.train_neural_spde import physical_samples, physical_scales, monthly_origins, write_json
    parser = argparse.ArgumentParser(description="Unified multiyear neural-SPDE/weather benchmark")
    parser.add_argument("--prepared", type=Path, default=ROOT/"data/processed/spde/multiyear_2015_2019")
    parser.add_argument("--models", type=Path, default=ROOT/"output/spde/neural_v1")
    parser.add_argument("--output", type=Path, default=ROOT/"output/spde/benchmark")
    parser.add_argument("--members", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--resume", action="store_true", help="Reuse matching baseline fits; recompute verification.")
    args = parser.parse_args()
    if args.members < 8:
        parser.error("Use at least eight members.")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError("Choose a new --output or --resume matching fitted models.")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = MultiYearWeather(args.prepared)
    models, scales = load_models(data, args.models, device), physical_scales(data)
    source_names = ("data/generate_data/benchmark_spde.py", "src/unified_benchmark.py", "src/neural_weather_spde.py",
        "src/weather_spde.py", "src/regional_spde.py",
        "data/generate_data/era5_synth.py", "data/generate_data/train_neural_spde.py")
    context = {"manifest_sha256": sha(args.prepared/"manifest.json"), "weights": {name: sha(args.models/name)
        for name in ("neural_best.pt", "constant_best.pt")}, "members": args.members, "seed": args.seed,
        "history": 24, "horizon": 72, "held_sites": data.held.tolist(), "split": data.manifest["split"],
        "source_sha256": {name: sha(ROOT/name) for name in source_names}}
    if args.resume and (out/"protocol.json").exists():
        if json.loads((out/"protocol.json").read_text())["context"] != context:
            raise ValueError("Benchmark data, weights, code or settings changed; use a new output.")
    print(f"Unified benchmark: {device}, {args.members} members, common 72h windows", flush=True)
    fixed = fixed_linear(models["fixed_spde"])
    ar_cache, fit_cache = out/"linear_baselines.npz", out/"baseline_fit.json"
    fit_context = {"manifest": context["manifest_sha256"], "adapter": sha(ROOT/"src/unified_benchmark.py")}
    if args.resume and ar_cache.exists() and fit_cache.exists():
        fitted = json.loads(fit_cache.read_text())
        if fitted["context"] != fit_context:
            raise ValueError("Baseline fitting context changed.")
        with np.load(ar_cache) as saved:
            spatial = {k: saved[k].copy() for k in ("f", "q", "initial", "drift")}
        spatial["observation_std"] = .025
    else:
        print("Fitting spatial AR(1) on visible training transitions", flush=True)
        spatial = fit_spatial_ar1(data)
        np.savez_compressed(ar_cache, **{k: spatial[k] for k in ("f", "q", "initial", "drift")})
        write_json(fit_cache, {"context": fit_context, "spatial_ar1": spatial["fit"], "test_selected": False})
    legacy_cache = out/"regional_reference_model.json"
    adapter_sha = hashlib.sha256(inspect.getsource(fit_monthly_legacy).encode()).hexdigest()
    if args.resume and legacy_cache.exists():
        legacy = json.loads(legacy_cache.read_text())
        if (legacy.get("data_manifest_sha256") != context["manifest_sha256"] or
            legacy.get("legacy_source_sha256") != sha(ROOT/"data/generate_data/era5_synth.py") or
            legacy.get("adapter_fit_source_sha256") != adapter_sha):
            raise ValueError("Legacy monthly fitting context changed.")
        for entry in legacy["training_quarters"]:
            if sha(ROOT/entry["file"]) != entry["sha256"]:
                raise ValueError("Legacy training quarter changed.")
    else:
        print("Refitting original monthly ARMA/Copula on the same training years", flush=True)
        legacy = fit_monthly_legacy(data, out, ROOT)
        legacy.update(data_manifest_sha256=context["manifest_sha256"], legacy_source_sha256=sha(ROOT/"data/generate_data/era5_synth.py"),
                      adapter_fit_source_sha256=adapter_sha)
        write_json(legacy_cache, legacy)
    thresholds = event_thresholds(data)
    protocol = {"context": context, "physical_scales_training_only": scales, "event_thresholds_training_only": thresholds,
        "conditional_models": list(CONDITIONAL), "calendar_only_models": list(SCENARIOS),
        "origin_rule": "Monthly day 15 00:30 UTC, continuous 72h; no future assimilation.",
        "conditional": "Exactly 24 preceding hours, only nonheld observations. Climate references ignore observed history.",
        "calendar_only": "No evaluation weather inputs. Neural/fixed/AR1 share training-only histories matched by month and UTC hour. Not yearlong/stationary generation.",
        "kalman_fixed_spde": "Same retained fixed coefficients; 24h Kalman state initialization with .025 observation std. Not a new Kalman parameter fit.",
        "spatial_ar1": "Training OLS persistence and empirical spatial/cross-field covariance; fixed .05 diagonal shrinkage and PSD floor. Discrete statistical baseline.",
        "legacy": {"source_site": legacy["source_site"], "scope": "Calendar-only, original monthly full AIC grid and sampler; true 10m wind from same training cache; single cell copied spatially with local clear sky.",
                   "preserved_limitations": "Per-day ARMA burn-in/restart, realized sample scaling and Gaussian CDF after heavy-tail innovations retained."},
        "limitations": ["2019 already inspected: unified development comparison, not a new blind test.",
            "12 windows per evaluation year, one Monte Carlo seed; members are not independent observed years.",
            "Equal variable/origin weights; GHI only clear>=50 W/m2, wind and temperature all valid hours.",
            "Old 2013 fits and short-record adapters excluded due to different data/preprocessing.",
            "Do not pool conditional and calendar-only rankings.",
            "Regional reanalysis only; no measured station, power or electrical-risk validation.",
            "Finite-member quantile error and window-edge event censoring remain."],
        "device": str(device), "torch": torch.__version__}
    write_json(out/"protocol.json", protocol)
    marginal, joint, structure, events, initialization = [], [], [], [], []
    for split in ("validation", "test"):
        selected = monthly_origins(data, split, 72)
        if len(selected) != 12*len(data.manifest["split"][split]):
            raise ValueError("Every evaluation month needs a full 72h window.")
        for origin in selected:
            indices, label, seed = np.arange(origin+1, origin+73), str(data.times[origin+1].date()), args.seed+int(origin)*17
            histories, training = prior_histories(data, origin, args.members, np.random.default_rng(seed), 72)
            initialization.extend({"split": split, "origin": label, "member": i, "training_history_end_utc": str(data.times[t])}
                                  for i, t in enumerate(training))
            for mode, names in (("conditional", CONDITIONAL), ("calendar_only", SCENARIOS)):
                for name in names:
                    rng = np.random.default_rng(seed)
                    if name in models:
                        paths = neural_paths(models[name], data, origin, args.members, device, seed,
                                             histories=None if mode == "conditional" else histories)
                        samples = physical_samples(data, paths, indices)
                    elif name in ("kalman_fixed_spde", "spatial_ar1"):
                        paths = linear_paths(data, fixed if name == "kalman_fixed_spde" else spatial, origin, 72,
                            args.members, rng, histories=None if mode == "conditional" else histories)
                        samples = physical_samples(data, paths, indices)
                    elif name == "legacy_arma_copula":
                        samples = legacy_paths(data, legacy, indices, args.members, seed)
                    else:
                        paths = (persistence(data, origin, 72, args.members, rng) if name == "persistence" else
                                 resample(data, origin, 72, args.members, rng, blocks=name == "history_blocks"))
                        samples = physical_samples(data, paths, indices)
                    for records, destination in zip(score_weather(data, indices, samples, scales, name, mode, label),
                                                    (marginal, joint, structure)):
                        destination.extend({"split": split, **r} for r in records)
                    events.append({"split": split, **event_score(data, indices, samples, thresholds, name, mode, label)})
                print(f"Scored {split} {label} {mode}: {len(names)} models", flush=True)
    frame = pd.DataFrame(marginal)
    frame.to_csv(out/"marginal_by_origin.csv", index=False)
    summary = frame.groupby(["split", "protocol", "model", "group", "variable", "lead_band"], sort=False).mean(numeric_only=True).reset_index()
    summary.to_csv(out/"marginal_summary.csv", index=False)
    for prefix, rows, keys in (("joint", joint, ["split", "protocol", "model", "group"]),
        ("structure", structure, ["split", "protocol", "model", "variable", "metric", "lag_hours"])):
        part = pd.DataFrame(rows)
        part.to_csv(out/f"{prefix}_by_origin.csv", index=False)
        part.groupby(keys, sort=False).mean(numeric_only=True).reset_index().to_csv(out/f"{prefix}_summary.csv", index=False)
    pd.DataFrame(events).to_csv(out/"events_by_origin.csv", index=False)
    pd.DataFrame(initialization).to_csv(out/"training_initializations.csv", index=False)
    intervals = paired_intervals(marginal, seed=args.seed)
    intervals.to_csv(out/"paired_intervals.csv", index=False)
    render_report(out, protocol, summary, pd.DataFrame(joint), pd.DataFrame(events), intervals)
    print(f"Unified benchmark complete: {out/'report.md'}", flush=True)


def render_report(out, protocol, marginal, joint, events, intervals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = marginal[(marginal.split == "test") & (marginal.group == "all") & (marginal.lead_band == "all")]
    for metric, filename, ylabel in (("crps_standardized", "comparison.png", "Standardized CRPS (lower better)"),
                                    ("coverage_90", "coverage.png", "90% interval coverage")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for mode, ax in zip(("conditional", "calendar_only"), axes):
            part = selected[selected.protocol == mode].pivot(index="model", columns="variable", values=metric)
            part.index = [NAMES[k] for k in part.index]
            part.plot.bar(ax=ax, rot=60)
            ax.set(xlabel="", ylabel=ylabel)
            ax.set_title(mode+": same 2019 windows", pad=12)
            ax.get_legend().remove()
            if metric == "coverage_90":
                ax.axhline(.9, color="black", ls="--"); ax.set_ylim(0, 1)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(out/filename, dpi=150); plt.close(fig)
    aggregate = selected.groupby(["protocol", "model"]).crps_standardized.mean()
    js = joint[(joint.split == "test") & (joint.group == "all")].groupby(["protocol", "model"]).spatial_variogram.mean()
    text = ["# 风光温统一比较：神经 SPDE 与基线\n",
        "本报告替代旧 2013 短记录 benchmark 和独立神经双模型比较。共同 ERA5 数据：2015—2017 年训练，2018 年验证，2019 年评价；36 网格中中央 4 格留出。神经和固定 SPDE 权重保持原样，其他基线只用训练数据拟合。",
        f"每年每月 15 日起报，连续 72 小时、{protocol['context']['members']} 个成员，不读未来天气。2019 年已在开发中查看，因此不是新的盲测；这些窗口也不是全年连续生成。\n",
        "## 本轮主要结果\n",
        f"- 条件预测平均标准化 CRPS：神经 SPDE {aggregate['conditional','neural_spde']:.4f}，固定 SPDE {aggregate['conditional','fixed_spde']:.4f}，空间 AR1 {aggregate['conditional','spatial_ar1']:.4f}。神经优于固定模型，但本轮边际误差没有超过空间 AR1。",
        f"- 不读测试天气的窗口生成：神经 SPDE {aggregate['calendar_only','neural_spde']:.4f}，原 ARMA/Copula {aggregate['calendar_only','legacy_arma_copula']:.4f}。神经边际误差更高；空间 VS 分别为 {js['calendar_only','neural_spde']:.4f} 和 {js['calendar_only','legacy_arma_copula']:.4f}，原单站共享方法的空间结构误差更大。",
        "- 不设一个掩盖取舍的总冠军。以下配对区间、分变量成绩、留出网格与联合轨迹指标一起判断；本轮不根据评价结果修改模型。\n",
        "## 共同协议\n",
        "conditional：可读取起报前 24 小时非留出网格天气。逐小时和历史块重采样是忽略实时历史的气候参考。",
        "calendar_only：不读取验证/测试天气；神经、固定 SPDE、空间 AR1 共享从训练年同月份、同 UTC 小时抽取的初始历史，原 ARMA/Copula 使用自身月模型先验。日历和晴空是已知量。两组结果不合并排名。\n"]
    for mode in ("conditional", "calendar_only"):
        table = selected[selected.protocol == mode].pivot(index="model", columns="variable", values="crps_standardized")
        held = marginal[(marginal.split == "test") & (marginal.protocol == mode) & (marginal.group == "held") &
                        (marginal.lead_band == "all")].groupby("model").crps_standardized.mean()
        text.extend([f"## 2019：{mode}\n", "| 方法 | GHI CRPS | 风速 CRPS | 温度 CRPS | 平均 | 留出平均 |", "|---|---:|---:|---:|---:|---:|"])
        for name, row in table.iterrows():
            text.append(f"| {NAMES[name]} | {row.ghi:.4f} | {row.wind_speed_100m:.4f} | {row.temp_air:.4f} | {row.mean():.4f} | {held[name]:.4f} |")
        text.append("\n按相同起报窗口配对，差值为神经减去基线，负值表示神经 CRPS 更低。描述性区间重采样 12 个窗口，不把成员当作实况重复。\n")
        subset = intervals[(intervals.split == "test") & (intervals.protocol == mode) & (intervals.group == "all")]
        for row in subset.itertuples():
            text.append(f"- 相对 {NAMES[row.baseline]}：改善 {row.skill:.1%}；差值 {row.mean_crps_difference:.4f}，95% 区间 [{row.ci_95_low:.4f}, {row.ci_95_high:.4f}]。")
        text.extend(["\n| 方法 | 轨迹 Energy | 时间 VS | 空间 VS |", "|---|---:|---:|---:|"])
        js = joint[(joint.split == "test") & (joint.protocol == mode) & (joint.group == "all")].groupby("model").mean(numeric_only=True)
        for name, row in js.iterrows():
            text.append(f"| {NAMES[name]} | {row.trajectory_energy:.4f} | {row.temporal_variogram:.4f} | {row.spatial_variogram:.4f} |")
    text.extend(["\n![统一 CRPS](comparison.png)\n", "![区间覆盖率](coverage.png)\n", "## 指标与方法边界\n",
        "CRPS、Energy、变差分数越低越好。CRPS/区间分数按训练年非留出物理尺度标准化，变量与窗口等权；GHI 只评价晴空≥50 W/m²，风速温度评价全部有效小时，所有模型使用同一目标遮罩。90% 覆盖率结合宽度、区间分数与有限成员分位误差判断。边际最优不代表时空结构、尾部风险全部最优。",
        "- 神经/固定 SPDE 为原多年同数据模型，没有根据本次评价重选权重。",
        "- Fixed SPDE + Kalman 只改变同一固定系数的状态初始化，不能据此声称比较了神经与 Kalman 参数训练。",
        "- 空间 AR1 为训练 OLS 时间系数和经验空间/跨变量创新协方差，加固定收缩与 PSD 修正；不是旧空间 OU 的换名。",
        "- 持续性加同季节、同时刻训练多步误差轨迹，具有不确定性。历史块保留连续 72 小时残差，逐小时重采样则打散时间联系。",
        f"- 原 ARMA/Copula 在可见参考格 {protocol['legacy']['source_site']} 按月重拟合同一训练期，使用真实 10 m 风速、原完整 AIC 搜索及采样器。单站输出被复制到区域，没有学习空间结构；仅参加 calendar_only。原 t 新息后正态 CDF、逐日滤波和样本缩放等做法保留，没有暗中修复。",
        "- 旧 2013 Kalman 模型、短记录适配和旧参数的共同/独立网格消融不混入排名，其训练与预处理不同。仍保留站点运行所需公共代码与模型。\n"])
    actual = events[(events.split == "test") & (events.protocol == "calendar_only") & (events.model == "neural_spde")]
    text.append(f"低风低光阈值只用训练期非留出区域：白天 Kc 25% 分位 {protocol['event_thresholds_training_only']['low_kc']:.3f}，风速 {protocol['event_thresholds_training_only']['low_wind']:.3f} m/s。2019 这些窗口有 {int(actual.observed_events.sum())} 个白天低风低光小时。事件持续时间不跨窗口或成员，边界有截断，有限事件不足以证明尾部已校准。")
    text.extend(["\n## 输出与复现\n",
        "marginal_by_origin/summary.csv：分变量、时效及全部/留出网格；joint_*.csv：整段轨迹；structure_*.csv：ACF、空间和跨变量相关；events_by_origin.csv：事件；paired_intervals.csv：配对区间；training_initializations.csv：无测试天气输入的训练历史来源。",
        "protocol.json 为数据、模型、源码与协议；linear_baselines.npz、baseline_fit.json、regional_reference_model.json 为可复用基线。没有额外保存大体积轨迹或电网算例。",
        "运行 `venv\\Scripts\\python.exe data/generate_data/benchmark_spde.py --resume`，复用匹配基线并重算。更换数据、权重、源码或设置时使用新 --output。训练入口只训练、验证选模并保存天气，比较统一走本入口。",
        "本轮仅验证区域天气，不验证真实站点局地、功率或潮流风险。一个评价年和固定 Monte Carlo 种子的局限仍存在。",
        "评分依据：[Gneiting & Raftery](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)、[Scheuerer & Hamill](https://repository.library.noaa.gov/view/noaa/22327/)。"])
    (out/"report.md").write_text("\n".join(text)+"\n", encoding="utf-8")
