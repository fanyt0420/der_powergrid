"""Training-only regional calibration and rolling-origin joint SPDE experiments.

This is a small-grid statistical weather-residual model. The coupling fit uses
filtered innovation proxies, not an exact joint maximum-likelihood estimator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import block_diag, solve_discrete_lyapunov
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtr

from src.weather_spde import (FieldParameters, JointWeatherSPDE, RectGrid,
                              gaussian_filter, kalman_update, evolution_operator,
                              ensemble_scores)

FIELDS = ("solar_logit", "wind_u_100m", "wind_v_100m", "temp_air")
DT = 3600.
SOLAR_CAP = 1.6


def regional_driver_model(report: dict, report_sha256: str) -> dict:
    """Retain fitted parameters and provenance needed by the final station layer."""
    keys = ("source", "context", "seed", "field_order", "grid_shape", "grid_x_m", "grid_y_m",
            "held_sites", "training", "validation", "test", "background", "selected_fits",
            "coupling_fit", "selected_coupling_alpha", "driving_correlation", "limitations")
    return {"schema": "regional-driver-v1", "status": report["status"],
            "derived_from_report_sha256": report_sha256, **{key: report[key] for key in keys}}


@dataclass
class RegionalData:
    times: pd.DatetimeIndex
    sites: pd.DataFrame
    grid: RectGrid
    raw: np.ndarray                 # time, field, node; transformed, unstandardized
    z: np.ndarray                   # same, training background removed and scaled
    background: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    clear: np.ndarray
    wind10: np.ndarray
    train_sites: np.ndarray
    held_sites: np.ndarray
    train_end: int
    validation_end: int
    metadata: dict


def calendar_design(times: pd.DatetimeIndex, xy: np.ndarray, origin: pd.Timestamp) -> np.ndarray:
    """Known calendar and coordinates only: no weather outcomes in the design."""
    hour = times.hour.to_numpy()+times.minute.to_numpy()/60
    day = (times-origin).total_seconds().to_numpy()/86400
    temporal = np.column_stack([np.ones(len(times)), day/30,
                                np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                                np.sin(4*np.pi*hour/24), np.cos(4*np.pi*hour/24)])
    design = np.empty((len(times), len(xy), 8))
    design[:, :, :6] = temporal[:, None, :]
    design[:, :, 6:] = xy[None, :, :]/100_000
    return design


def load_region(prepared: Path, train_days: int = 42, validation_days: int = 10) -> RegionalData:
    weather = pd.read_csv(prepared/"weather_hourly.csv.gz")
    weather["timestamp_utc"] = pd.to_datetime(weather.timestamp_utc, utc=True)
    sites = pd.read_csv(prepared/"sites.csv").sort_values(["y_m", "x_m"]).reset_index(drop=True)
    grid = RectGrid(np.sort(sites.x_m.unique()), np.sort(sites.y_m.unique()))
    if len(sites) != grid.size or min(len(grid.x), len(grid.y)) < 4:
        raise ValueError("Regional experiment needs a complete grid, at least 4x4.")
    times = pd.DatetimeIndex(sorted(weather.timestamp_utc.unique()))
    if not np.all(np.diff(times.asi8) == int(DT*1e9)):
        raise ValueError("Prepared hours must be complete and regularly spaced.")
    def values(column):
        return weather.pivot(index="timestamp_utc", columns="site_id", values=column).reindex(
            index=times, columns=sites.site_id).to_numpy(float)
    kc = values("clear_sky_index")
    solar = logit(np.clip(kc/SOLAR_CAP, .001, .999))
    raw = np.stack([solar, values("wind_u_100m"), values("wind_v_100m"), values("temp_air")], axis=1)
    clear = values("clear_sky_ghi")
    if not np.isfinite(raw[:, 1:]).all() or not np.isfinite(clear).all():
        raise ValueError("Missing wind/temperature/clear sky in prepared data.")
    train_end, val_end = train_days*24, (train_days+validation_days)*24
    if not 7*24 <= train_end < val_end < len(times):
        raise ValueError("Need >=7 training days, positive validation and a future test period.")
    nx, ny = len(grid.x), len(grid.y)
    held = np.array([iy*nx+ix for iy in (ny//2-1, ny//2) for ix in (nx//2-1, nx//2)])
    train = np.setdiff1d(np.arange(grid.size), held)
    design = calendar_design(times, grid.xy, times[0])
    background, scale, coefficients = np.empty_like(raw), [], []
    for i in range(4):
        # Use take rather than mixed advanced indexing, retaining time/site order.
        target = np.take(raw[:train_end, i], train, axis=1).reshape(-1)
        x = np.take(design[:train_end], train, axis=1).reshape(-1, 8)
        valid = np.isfinite(target)
        coef = np.linalg.lstsq(x[valid], target[valid], rcond=None)[0]
        background[:, i] = np.einsum("tnk,k->tn", design, coef)
        residual = np.take(raw[:train_end, i]-background[:train_end, i], train, axis=1)
        scale.append(float(np.nanstd(residual)))
        coefficients.append(coef)
    scale = np.array(scale)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Degenerate training residual scale.")
    metadata = json.loads((prepared/"provenance.json").read_text(encoding="utf-8"))
    kc_train = np.take(kc[:train_end], train, axis=1)
    finite = kc_train[np.isfinite(kc_train)]
    metadata["solar_transform"] = {"formula": "logit(clip(Kc/1.6, 0.001, 0.999)); missing when clear GHI<50 W/m2",
                                   "training_clipped_fraction": float(np.mean((finite <= .001*SOLAR_CAP)
                                                                            | (finite >= .999*SOLAR_CAP))),
                                   "inverse": "Kc=1.6*expit(background+scale*z); empirical marginals not guaranteed"}
    return RegionalData(times, sites, grid, raw, (raw-background)/scale[None, :, None],
                        background, scale, np.array(coefficients), clear, values("wind_speed"),
                        train, held, train_end, val_end, metadata)


def fit_field(data: RegionalData, field: int, *, transport: bool, max_evaluations: int = 220,
              warm_start: dict | None = None) -> dict:
    """Sum 12 separate daily Kalman likelihoods, all from the training period.

    Splitting speeds a research pilot. Each day starts from the stationary prior;
    this is a composite likelihood, not the uninterrupted 42-day likelihood.
    """
    y = np.take(data.z[:data.train_end, field], data.train_sites, axis=1)
    h = np.eye(data.grid.size)[data.train_sites]
    days = np.unique(np.linspace(0, data.train_end//24-1, 12).astype(int))
    chunks = [y[24*d:24*(d+1)] for d in days]
    # v/10 makes numerical search scales comparable to log parameters.
    bounds = ([(-2., 2.), (-2., 2.), (np.log(10), np.log(200_000))] if transport else [])
    bounds += [(np.log(1/(5*86400)), np.log(1/3600)), (np.log(.0002), np.log(.2)),
               (np.log(1000), np.log(150_000)), (np.log(.015), np.log(.8))]
    base = np.log([1/86400, .008, 30000, .08])
    start = np.r_[[.2, .05, np.log(5000)], base] if transport else base
    if warm_start and transport:
        p = warm_start["parameters"]
        velocity = ([p["velocity_x_m_s"]/10, p["velocity_y_m_s"]/10,
                     np.log(p["diffusivity_m2_s"])] if warm_start.get("transport") else [.2, .05, np.log(5000)])
        start = np.r_[velocity, np.log([p["decay_per_s"], p["noise_std_per_sqrt_s"],
                                        p["noise_length_m"], warm_start["observation_std"]])]
    def unpack(theta):
        vx, vy, diffusion = (theta[0]*10, theta[1]*10, np.exp(theta[2])) if transport else (0., 0., 0.)
        damping, sigma, length, obs = np.exp(theta[3:] if transport else theta)
        return FieldParameters(FIELDS[field], float(vx), float(vy), float(diffusion),
                               float(damping), float(sigma), float(length)), float(obs)
    def objective(theta):
        try:
            p, obs = unpack(theta)
            f, g, initial = JointWeatherSPDE(data.grid, [p], DT).dense_single_field()
            return sum(gaussian_filter(chunk, h, f, g@g.T, initial, obs, store_history=False)["nll"]
                       for chunk in chunks)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1e20
    # Powell handles the upwind operator's non-differentiability at v=0.
    method = "Powell" if transport else "L-BFGS-B"
    options = ({"maxfev": max_evaluations, "ftol": 2e-4, "xtol": 2e-3} if transport else
               {"maxfun": max_evaluations, "maxiter": 50, "ftol": 1e-6, "eps": 1e-4})
    result = minimize(objective, start, method=method, bounds=bounds, options=options)
    p, obs = unpack(result.x)
    if not np.isfinite(result.fun) or result.fun > 1e19:
        raise RuntimeError(f"No finite fit for {FIELDS[field]}.")
    return {"parameters": asdict(p), "observation_std": obs, "training_composite_nll": float(result.fun),
            "optimizer_converged": bool(result.success), "optimizer_message": str(result.message),
            "evaluations": int(result.nfev), "training_days_used": days.tolist(),
            "transport": transport,
            "parameters_at_bounds": [i for i, (x, (lo, hi)) in enumerate(zip(result.x, bounds, strict=True))
                                     if min(abs(x-lo), abs(x-hi)) < 2e-3]}


def filtered_driver_correlation(data: RegionalData, fits: list[dict]) -> tuple[np.ndarray, dict]:
    """Infer approximate white drivers from marginal filtered means, training only.

    Missing locations are conditional estimates, never their held-out targets.
    Filtering and inverse smoothing can bias these proxies; record that limit.
    """
    innovations = []
    for i, fit in enumerate(fits):
        model = JointWeatherSPDE(data.grid, [FieldParameters(**fit["parameters"])], DT)
        f, g, initial = model.dense_single_field()
        y = np.take(data.z[:data.train_end, i], data.train_sites, axis=1)
        filtered = gaussian_filter(y, np.eye(data.grid.size)[data.train_sites], f, g@g.T,
                                   initial, fit["observation_std"])["means"]
        p = model.fields[0]
        # Invert the very same implicit propagator and Helmholtz noise smoother.
        m = np.eye(data.grid.size)-DT*evolution_operator(data.grid, p).toarray()
        lap = evolution_operator(data.grid, FieldParameters("lap", diffusivity_m2_s=1, decay_per_s=0)).toarray()
        s = np.eye(data.grid.size)-p.noise_length_m**2*lap
        eps = (filtered[1:]@m.T-filtered[:-1])@s.T
        eps *= model._norms[0]/(p.noise_std_per_sqrt_s*np.sqrt(DT))
        innovations.append(np.take(eps, data.train_sites, axis=1))
    proxies = np.stack(innovations, axis=-1)
    # Daytime complete cases in consecutive hours; no solar imputation at night.
    daytime = np.isfinite(np.take(data.z[:data.train_end, 0], data.train_sites, axis=1))
    valid = daytime[1:] & daytime[:-1] & np.isfinite(proxies).all(axis=-1)
    rows = proxies[valid]
    correlation = np.corrcoef(rows.T)
    if len(rows) < 100 or not np.isfinite(correlation).all():
        raise ValueError("Too few complete training innovations for a joint fit.")
    return correlation, {"method": "sample correlation of inverse-filtered marginal filtered-mean proxies",
                         "rows": len(rows), "fit_period": "training only; consecutive daytime hours",
                         "not_exact_joint_mle": True,
                         "limitation": "proxy shrinkage and inverse smoothing bias; daylight fit used for all hours"}


def system(data: RegionalData, fits: list[dict], correlation: np.ndarray,
           mode: str = "joint_spde") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fields = [FieldParameters(**fit["parameters"]) for fit in fits]
    model = JointWeatherSPDE(data.grid, fields, DT, correlation)
    f, g, initial = model.dense_joint()
    q = g@g.T
    n = data.grid.size
    if mode in ("independent_sites", "common_weather"):
        # Matched average stationary marginal variances. These are structural
        # ablations, not independently optimized competitors.
        rho = [float(np.mean(f[i*n:(i+1)*n, i*n:(i+1)*n].sum(axis=1))) for i in range(4)]
        variance = [float(np.trace(initial[i*n:(i+1)*n, i*n:(i+1)*n])/n) for i in range(4)]
        amplitude = np.sqrt(np.array(variance)*(1-np.array(rho)**2))
        f = block_diag(*[r*np.eye(n) for r in rho])
        spatial = np.eye(n) if mode == "independent_sites" else np.ones((n, n))
        q = np.kron(correlation*amplitude[:, None]*amplitude[None, :], spatial)
        initial = solve_discrete_lyapunov(f, q)
        initial = (initial+initial.T)/2
    elif mode != "joint_spde":
        raise ValueError(f"Unknown mode {mode}")
    return f, q, initial, np.repeat([fit["observation_std"] for fit in fits], n)


def covariance_factor(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((covariance+covariance.T)/2)
    if values.min() < -1e-7:
        raise ValueError("Predictive covariance is not positive semidefinite.")
    return vectors*np.sqrt(np.maximum(values, 0))


def physical(data: RegionalData, z: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Accept (..., time, field, node); output physical nonnegative GHI/speed."""
    raw = z*data.scale[:, None]+data.background[indices]
    return {"ghi": SOLAR_CAP*expit(raw[..., 0, :])*data.clear[indices],
            "wind_u_100m": raw[..., 1, :], "wind_v_100m": raw[..., 2, :],
            "wind_speed_100m": np.hypot(raw[..., 1, :], raw[..., 2, :]),
            "temp_air": raw[..., 3, :]}


def _marginal_scores(truth, means, variances):
    good = np.isfinite(truth)
    sd = np.sqrt(np.maximum(variances[good], 1e-14))
    residual = truth[good]-means[good]
    z = residual/sd
    crps = sd*(z*(2*ndtr(z)-1)+2*np.exp(-.5*z*z)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    return {"count": int(good.sum()), "standardized_rmse": float(np.sqrt(np.mean(residual**2))),
            "standardized_crps": float(np.mean(crps)), "coverage_90": float(np.mean(abs(z) <= 1.6448536269514722)),
            "mean_width_90": float(np.mean(2*1.6448536269514722*sd))}


def evaluate(data: RegionalData, matrices: tuple, start: int, stop: int, *, members: int = 96,
             horizon: int = 6, seed: int = 20261022, store_paths: bool = False) -> tuple[dict, dict]:
    """Rolling origins, nonoverlapping 1..6-hour forecasts, no future assimilation.

    A 48-hour pre-start warmup uses only nonheld cells. Subsequent forecasts may
    use newly arrived past data. Parameters remain frozen throughout validation/test.
    """
    f, q, initial, obs = matrices
    n, dim = data.grid.size, len(f)
    observed_indices = np.concatenate([i*n+data.train_sites for i in range(4)])
    h = np.eye(dim)[observed_indices]
    obs_train = obs[observed_indices]
    mean, covariance = np.zeros(dim), initial.copy()
    factor_q = covariance_factor(q)
    rng = np.random.default_rng(seed)
    paths, means, variances, targets, leads = [], [], [], [], []
    first_posterior = None
    for t in range(max(0, start-48), stop-1):
        mean = f@mean
        covariance = f@covariance@f.T+q
        row = data.z[t].reshape(-1)[observed_indices]
        mean, covariance, _ = kalman_update(mean, covariance, row, h, obs_train)
        if t+1 < start or (t+1-start) % horizon or t+1+horizon > stop:
            continue
        if first_posterior is None:
            first_posterior = (mean.copy(), covariance.copy())
        state = mean+rng.standard_normal((members, dim))@covariance_factor(covariance).T
        predicted, cov = mean.copy(), covariance.copy()
        for lead in range(1, horizon+1):
            state = state@f.T+rng.standard_normal((members, dim))@factor_q.T
            predicted = f@predicted
            cov = f@cov@f.T+q
            # Validation targets are reanalysis values; represent unresolved
            # model discrepancy using the fitted observation-error term.
            observed = state+rng.standard_normal((members, dim))*obs
            paths.append(observed.reshape(members, 4, n))
            means.append(predicted.reshape(4, n))
            variances.append((np.diag(cov)+obs**2).reshape(4, n))
            targets.append(t+lead)
            leads.append(lead)
    indices = np.array(targets)
    if not len(indices):
        raise ValueError("Evaluation period must include at least one complete forecast horizon.")
    paths = np.stack(paths, axis=1)
    means, variances = np.array(means), np.array(variances)
    truth = data.z[indices]
    scores = {"forecast_hours": len(indices), "origins": len(indices)//horizon, "horizon_hours": horizon,
              "members": members, "fields": {}}
    for i, name in enumerate(FIELDS):
        scores["fields"][name] = {"all_sites": _marginal_scores(truth[:, i], means[:, i], variances[:, i]),
                                   "held_sites": _marginal_scores(truth[:, i, data.held_sites],
                                                                  means[:, i, data.held_sites],
                                                                  variances[:, i, data.held_sites]),
                                   "spatial_scores": ensemble_scores(truth[:, i], paths[:, :, i], data.grid.xy)}
    energy = []
    for k in range(len(indices)):
        valid = np.isfinite(truth[k].reshape(-1))
        x, y = paths[:, k].reshape(members, -1)[:, valid], truth[k].reshape(-1)[valid]
        energy.append((np.linalg.norm(x-y, axis=1).mean()
                       -.5*np.linalg.norm(x-np.roll(x, 1, axis=0), axis=1).mean())/np.sqrt(valid.sum()))
    scores["joint_energy_score_per_sqrt_dimension"] = float(np.mean(energy))
    scores["mean_spatial_variogram"] = float(np.mean([v["spatial_scores"]["variogram_score_p_half"]
                                                     for v in scores["fields"].values()]))
    truth_p, path_p = physical(data, truth, indices), physical(data, paths, indices)
    scores["physical"] = {}
    for name in ("ghi", "wind_speed_100m", "temp_air"):
        target, sample = truth_p[name], path_p[name]
        # At night solar z is missing; evaluate physical GHI only where clear>=50.
        valid = np.isfinite(target)
        lo, hi = np.quantile(sample, [.05, .95], axis=0)
        scores["physical"][name] = {"rmse": float(np.sqrt(np.mean((target[valid]-sample.mean(axis=0)[valid])**2))),
                                    "coverage_90": float(np.mean((target[valid] >= lo[valid]) & (target[valid] <= hi[valid]))),
                                    "units": {"ghi": "W/m2 (daytime)", "wind_speed_100m": "m/s", "temp_air": "degC"}[name]}
    # A direct compound-weather diagnostic, with thresholds frozen on training
    # cells/hours. The same 32-cell regional average defines all event targets.
    cells = data.train_sites
    train_solar = data.raw[:data.train_end, 0][:, cells]
    daylight_train = np.isfinite(train_solar).all(axis=1)
    train_kc = (SOLAR_CAP*expit(train_solar[daylight_train])).mean(axis=1)
    train_speed = np.hypot(data.raw[:data.train_end, 1][:, cells],
                           data.raw[:data.train_end, 2][:, cells])[daylight_train].mean(axis=1)
    solar_threshold, wind_threshold = float(np.quantile(train_kc, .25)), float(np.quantile(train_speed, .25))
    daylight = np.isfinite(data.raw[indices, 0][:, cells]).all(axis=1)
    true_kc = (SOLAR_CAP*expit(data.raw[indices, 0][:, cells]))[daylight].mean(axis=1)
    true_speed = truth_p["wind_speed_100m"][:, cells][daylight].mean(axis=1)
    raw_paths = paths*data.scale[:, None]+data.background[indices]
    sample_kc = (SOLAR_CAP*expit(raw_paths[:, :, 0, :][:, :, cells]))[:, daylight].mean(axis=2)
    sample_speed = path_p["wind_speed_100m"][:, :, cells][:, daylight].mean(axis=2)
    event = (true_kc < solar_threshold) & (true_speed < wind_threshold)
    probabilities = ((sample_kc < solar_threshold) & (sample_speed < wind_threshold)).mean(axis=0)
    scores["compound_low_solar_low_wind"] = {
        "definition": "daytime 32-cell mean Kc and 100m speed both below respective training daytime 25th percentiles",
        "solar_kc_threshold": solar_threshold, "wind_speed_threshold_m_s": wind_threshold,
        "daytime_hours": int(len(event)), "observed_events": int(event.sum()),
        "observed_frequency": float(event.mean()), "mean_predicted_probability": float(probabilities.mean()),
        "brier_score": float(np.mean((probabilities-event)**2)),
        "contains_events_and_nonevents": bool(event.any() and not event.all()),
        "interpretation": "An event-free test cannot establish joint-tail calibration; Brier alone can reward zero forecasts."}
    extra = {"indices": indices, "mean": means, "variance": variances, "first_posterior": first_posterior,
             "lead_hours": np.array(leads)}
    if store_paths:
        extra["predictive_paths"] = paths
    return scores, extra


def sample_contiguous(data: RegionalData, matrices: tuple, posterior: tuple, start: int,
                      hours: int = 24, members: int = 96, seed: int = 20261023) -> dict:
    """A single-origin, continuous weather ensemble, excluding discrepancy noise."""
    f, q, _, _ = matrices
    mean, cov = posterior
    rng = np.random.default_rng(seed)
    state = mean+rng.standard_normal((members, len(f)))@covariance_factor(cov).T
    factor = covariance_factor(q)
    paths = []
    for _ in range(hours):
        state = state@f.T+rng.standard_normal(state.shape)@factor.T
        paths.append(state.reshape(members, 4, data.grid.size))
    paths = np.stack(paths, axis=1)
    indices = np.arange(start, start+hours)
    return {**physical(data, paths, indices), "latent": paths, "timestamps": data.times[indices].asi8,
            "site_ids": data.sites.site_id.to_numpy(str)}
