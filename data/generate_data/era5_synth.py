"""ARIMA(kt) + Gaussian-copula synthetic weather generator for the ERA5 cases.

Why this exists
---------------
``ieee13_era5.py`` in its original form *replays* observed days: one run = one real
day, and the only "randomness" is which historical day you drew. That has two hard
limits:

1. The scenario set cannot be larger than the observed record, and the tails are
   exactly as sparse as history (a 1-in-30-years cloudy spell cannot be *sampled*,
   only *replayed* once).
2. Because the marginals are empirical, you cannot ask "what if there were 300 days
   like this summer".

This module adds a generative layer on top of the same ERA5 record, so the scenario
set can be enlarged and its tail behaviour controlled, while pvlib stays the
deterministic weather -> power converter.

The model
---------
Everything is fitted **per calendar month** (12 groups), because both the diurnal
cycle and the dependence structure are strongly seasonal.

1. **Clearness index** ``kt = GHI / GHI_clearsky`` (pvlib ``ineichen``, the same
   model used when sampling, so fit and generation are consistent). ``kt`` removes
   the deterministic solar geometry, which is what makes a stationary time-series
   model admissible at all. It is undefined at night (0/0), so night hours are
   ``NaN`` -- and ``NaN`` is passed to ``statsmodels``' state-space ARIMA, whose
   Kalman filter skips them. That matters: concatenating the daylight segments and
   ignoring the overnight gap injects a spurious lag-1 pair at every day boundary
   and biases the autocorrelation downward (verified on synthetic data with a known
   AR(1): 0.570 with the NaN mask vs 0.563 without, true value 0.6).

2. **Two-level temporal model on normal scores.** ``kt`` is still not stationary
   inside the daylight window (air-mass dependence), so it is first standardised by
   its (month, hour) climatology, then mapped to normal scores
   ``z = Phi^-1(rank/(n+1))``. Fitting the ARMA on the normal scores instead of on
   ``kt`` itself is what keeps the *empirical* marginal: a Gaussian-innovation ARMA
   fitted on raw ``kt`` would generate Gaussian, hence negative, clearness.
   The temporal model is split in two, because a single ARMA over a night-masked
   series can only see within-day dependence and would leave consecutive days
   independent -- which is what made the first version of this module collapse the
   day-to-day variance of temperature (std 9.9 -> 8.3 degC) and, much worse, of wind
   (daily mean std 2.18 -> 0.91 m/s, i.e. 42% of the real spread):
     * **day level**: VAR(1) on the 4-vector of daily mean normal scores. This is
       the synoptic persistence (real cloudy spells last several days) *and* the
       day-scale cross-dependence, in one model;
     * **within day**: per-variable ARMA(p,q) selected by AIC, on the deviation from
       that day's mean, fitted on the night-masked series.
   Orders are chosen per month per variable from
   ``((1,0,0),(2,0,0),(3,0,0),(4,0,0),(1,0,1),(2,0,1),(3,0,1),(0,0,1))``.

3. **Gaussian copula** across ``(kt, temp_air, wind_speed_100m, wind_speed)``, in
   the same normal-score space. Rather than imposing the correlation on the filtered
   *outputs* (which the ARMA filters would then distort), it is imposed on the
   *innovations* and corrected: if variable ``u`` has MA(infinity) coefficients
   ``psi_u``, feeding innovations with correlation ``R_e`` produces outputs with
   ``R_uv = [R_e]_uv * <psi_u,psi_v> / (|psi_u||psi_v|)``, so ``R_e = R / C`` with
   that cosine-similarity matrix ``C``. ``R_e`` is then projected onto the nearest
   positive-definite matrix with unit diagonal. The realised output correlation is
   reported back by ``--validate`` so the correction can be checked, not assumed.
   Sampling is *conditional* in the sense that the ARIMA supplies the kt margin and
   the other three follow the fitted dependence; at night ``kt`` is undefined, so its
   innovation is drawn independently (the day-level VAR still carries the synoptic
   state, so night weather is not decoupled from the day).

4. **Back-transform**: normal score -> empirical quantile (per month, 501-point
   grid, tails clipped) -> anomaly -> ``mu[month,hour] + sigma[month,hour] * anomaly``.
   ``GHI = kt * GHI_clearsky``; ``DNI``/``DHI`` come from pvlib's ERBS decomposition
   so the three components stay mutually consistent.

Usage
-----
    # fit (needs the multi-year ERA5 cache: data/raw/era5/<site>_<year>.csv)
    python data/generate_data/era5_synth.py --fit --site atlanta
    python data/generate_data/era5_synth.py --fit --site atlanta --fit-years 1990 2014

    # validate against held-out years
    python data/generate_data/era5_synth.py --validate --site atlanta

    # feed the generator into the QSTS case
    python data/generate_data/ieee13_era5.py --weather synth --scenarios 200 --start 2022-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import kurtosis, norm, wasserstein_distance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import pvlib  # noqa: E402
from pvlib.location import Location  # noqa: E402

from data.generate_data.ieee13_era5 import ERA5_CACHE, SITES  # noqa: E402

warnings.filterwarnings("ignore")

SYNTH_DIR = ROOT / "output" / "synth"

# Variable order of the copula. kt first: it is the variable the ARIMA drives.
VARS = ("kt", "temp_air", "wind_speed_100m", "wind_speed")

# Below this clearsky GHI (W/m2) the ratio GHI/GHI_clear is noise-dominated and kt
# is left undefined. 50 W/m2 is ~5% of peak, i.e. solar elevation of a few degrees.
KT_MIN_CLEAR = 50.0
# Cloud enhancement can push kt above 1; 1.25 is the usual plotting ceiling.
KT_MAX = 1.25

# Empirical quantile grid stored per (month, variable). Tails outside it are clipped,
# which caps the synthetic record's extremes at roughly the observed range.
QPROBS = np.linspace(0.0005, 0.9995, 501)

ARIMA_ORDERS = ((1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0),
                (1, 0, 1), (2, 0, 1), (3, 0, 1), (0, 0, 1))

DEFAULT_FIT_YEARS = (1990, 2014)
DEFAULT_VAL_YEARS = (2015, 2024)
SYNTH_SEED = 20251014


# ---------------------------------------------------------------------------
# Data loading / clearness index
# ---------------------------------------------------------------------------

def load_years(site: str, years: tuple[int, int]) -> pd.DataFrame:
    frames = []
    missing = []
    for year in range(years[0], years[1] + 1):
        path = ERA5_CACHE / f"{site}_{year}.csv"
        if not path.exists():
            missing.append(year)
            continue
        frames.append(pd.read_csv(path, index_col=0, parse_dates=True))
    if not frames:
        raise SystemExit(f"no ERA5 years for {site} in {years}; run _fetch_years.py first")
    if missing:
        print(f"warning           : {site} missing {len(missing)} year(s) in {years}: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    frame = pd.concat(frames).sort_index()
    return frame[~frame.index.duplicated(keep="first")].astype(float)


def _clearsky(site: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Instantaneous clear-sky GHI / zenith, plus the hour-average clear-sky GHI.

    ERA5 (and the Open-Meteo archive endpoint that serves it) reports shortwave
    radiation as the *average over the preceding hour*, so the value stamped h covers
    [h-1, h], while pvlib's clearsky is instantaneous at the stamp. Dividing the one
    by the other gives a clearness index that is not physically meaningful near the
    ends of the day: at Atlanta it runs to 1.11 at 19:00 and 1.22 at 20:00, which is
    above the Linke-turbidity clearsky it is normalised by. Worse, it makes the
    synthetic record collapse to zero about an hour before the real one does, which
    shifts the evening PV ramp -- the one that sets the duck curve.

    kt is therefore defined against ``ghi_clear_avg``, the clearsky integrated over
    [h-1, h] by a 2-point midpoint rule (samples at h-45min and h-15min). On the
    same hours that gives kt of 0.80 and 0.85, and the daylight window extends by
    roughly an hour at each end, matching ERA5.
    """
    info = SITES[site]
    # Fixed-offset clock, not the civil zone: see the note at SITES in ieee13_era5.py.
    tz = info["solar_tz"]
    location = Location(info["lat"], info["lon"], tz=tz)
    times = index.tz_localize(tz) if index.tz is None else index
    sky = location.get_clearsky(times, model="ineichen")
    pos = location.get_solarposition(times)

    # Midpoint rule on [h-1, h]: clearsky is smooth in time, so two interior samples
    # integrate it to well under a percent, and unlike a trapezoid on (h-1, h) it does
    # not overshoot where the curve is convex at sunrise and sunset.
    sub_offsets = (np.timedelta64(-45, "m"), np.timedelta64(-15, "m"))
    sub = pd.DatetimeIndex(np.concatenate([index.values + off for off in sub_offsets]))
    sub_sky = location.get_clearsky(sub.tz_localize(tz), model="ineichen")["ghi"].to_numpy(float)
    ghi_clear_avg = sub_sky.reshape(len(sub_offsets), len(index)).mean(axis=0)

    return pd.DataFrame({"ghi_clear": sky["ghi"].to_numpy(float),
                         "ghi_clear_avg": ghi_clear_avg,
                         "zenith": pos["apparent_zenith"].to_numpy(float)},
                        index=index)


def clearness(frame: pd.DataFrame, site: str) -> pd.DataFrame:
    """Attach ghi_clear / zenith / kt (NaN at night) to an ERA5 frame."""
    out = frame.copy()
    sky = _clearsky(site, frame.index)
    out["ghi_clear"] = sky["ghi_clear"]
    out["ghi_clear_avg"] = sky["ghi_clear_avg"]
    out["zenith"] = sky["zenith"]
    # Daylight is decided on the hour-average clearsky, because that is the quantity
    # ERA5's own GHI is comparable with; on the instantaneous one the last daylight
    # hour of the real record would be discarded as night.
    day = sky["ghi_clear_avg"] > KT_MIN_CLEAR
    kt = np.where(day, out["ghi"].to_numpy(float) / np.maximum(sky["ghi_clear_avg"], 1e-9), np.nan)
    out["kt"] = np.clip(kt, 0.0, KT_MAX)
    return out


# ---------------------------------------------------------------------------
# Normal scores / empirical margins
# ---------------------------------------------------------------------------

def normal_scores(x: np.ndarray) -> np.ndarray:
    """Rank-based probability integral transform -> standard normal scores."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    z = np.full(x.shape, np.nan)
    ranks = pd.Series(x[ok]).rank(method="average").to_numpy()
    n = ok.sum()
    u = ranks / (n + 1.0)
    z[ok] = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    return z


def quantile_grid(values: np.ndarray) -> list[float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 50:
        return [0.0] * len(QPROBS)
    return np.quantile(v, QPROBS).round(6).tolist()


def quantile_apply(grid: list[float], u: np.ndarray) -> np.ndarray:
    """Inverse empirical CDF: u in (0,1) -> value, linear between grid points."""
    g = np.asarray(grid, float)
    return np.interp(np.clip(u, QPROBS[0], QPROBS[-1]), QPROBS, g)


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def _arma_polys(res) -> tuple[np.ndarray, np.ndarray]:
    """statsmodels fit -> (ar, ma) lag polynomials for arma_generate_sample."""
    ar = np.r_[1.0, -np.asarray(res.arparams, float)]
    ma = np.r_[1.0, np.asarray(res.maparams, float)]
    return ar, ma


def _arma_unit_sigma(ar: np.ndarray, ma: np.ndarray) -> float:
    """Innovation sigma that makes an ARMA(ar,ma) have unit variance."""
    from statsmodels.tsa.arima_process import arma2ma
    psi = arma2ma(ar, ma, lags=200)
    return float(1.0 / np.sqrt(np.sum(psi ** 2)))


def fit_arima(z: np.ndarray, orders=ARIMA_ORDERS) -> dict:
    """Select an ARIMA order by AIC on a series that may contain NaN gaps."""
    from statsmodels.tsa.arima.model import ARIMA
    best = None
    # The module-level filterwarnings("ignore") does NOT silence statsmodels here.
    # Importing statsmodels *prepends* ("always", ModelWarning) to the filter list,
    # and EstimationWarning -- the "non-stationary starting autoregressive
    # parameters" notice -- subclasses ModelWarning, so matching stops on statsmodels'
    # own entry before it ever reaches ours. catch_warnings + simplefilter replaces
    # the list outright, which is the only way to get ahead of it.
    # These notices are advisory: start_params is only the optimiser's seed, a bad
    # seed costs iterations rather than accuracy, and a candidate that starts badly
    # simply loses the AIC comparison. At 8 orders x 4 variables x 12 months they
    # otherwise bury the per-month diagnostics that fit() prints.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in orders:
            try:
                res = ARIMA(np.asarray(z, float), order=order, trend="n").fit(method_kwargs={"maxiter": 200})
            except Exception:  # noqa: BLE001
                continue
            aic = float(res.aic)
            if not np.isfinite(aic):
                continue
            if best is None or aic < best["aic"] - 1e-9:
                ar, ma = _arma_polys(res)
                best = {"order": list(order), "aic": round(aic, 2), "ar": ar.tolist(), "ma": ma.tolist(),
                        # Innovate with the sigma that makes the ARMA unit-variance, so the
                        # generated within-day series has exactly the fitted spread once it
                        # is multiplied back by sd. The fit's own sigma2 is not used: this
                        # way the two-level recombination has the right total variance by
                        # construction instead of only approximately.
                        "sigma": round(_arma_unit_sigma(ar, ma), 8)}
    if best is None:
        best = {"order": [0, 0, 0], "aic": None, "ar": [1.0], "ma": [1.0], "sigma": 1.0}
    return best


def _lag1(series: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D array, ignoring NaN pairs."""
    s = np.asarray(series, float)
    a, b = s[:-1], s[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float("nan")
    a, b = a[ok], b[ok]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _acf1_model(ar: np.ndarray, ma: np.ndarray) -> float:
    """Lag-1 autocorrelation implied by an ARMA, for checking the AIC pick."""
    from statsmodels.tsa.arima_process import arma2ma
    psi = np.asarray(arma2ma(np.asarray(ar, float), np.asarray(ma, float), lags=400), float)
    denom = np.sum(psi ** 2)
    return float(np.sum(psi[1:] * psi[:-1]) / denom) if denom > 1e-12 else float("nan")


def _within_day_diff(z: np.ndarray, same_day: np.ndarray) -> np.ndarray:
    """Consecutive differences of ``z``, dropping pairs that cross a day boundary."""
    z = np.asarray(z, float)
    ok = np.isfinite(z)
    d = np.full(z.shape, np.nan)
    d[1:] = np.where(ok[1:] & ok[:-1] & same_day[1:], z[1:] - z[:-1], np.nan)
    return d[~np.isnan(d)]


def _innovation_kurtosis(gamma_delta: float, ar: np.ndarray, ma: np.ndarray,
                         lags: int = 400) -> float:
    """Excess kurtosis of the INNOVATION given that of the observed increment.

    For a linear filter y = sum_j psi_j e_{t-j} driven by iid e with excess kurtosis
    g, the fourth cumulant is additive and scales as c^4, so

        g(y) = g(e) * sum_j psi_j^4 / (sum_j psi_j^2)^2 .

    The increment is another linear filter with weights c_j = psi_j - psi_{j-1}
    (c_0 = psi_0 = 1), so the measured increment kurtosis understates the innovation
    kurtosis by exactly that ratio and we divide it back out. Needed because the
    intermittency of the *increments* -- ramps and drop events -- is what the grid
    cares about, while the free parameter lives on the innovation.
    """
    from statsmodels.tsa.arima_process import arma2ma
    psi = np.asarray(arma2ma(np.asarray(ar, float), np.asarray(ma, float), lags=lags), float)
    c = np.diff(np.concatenate([[0.0], psi]))  # c_j = psi_j - psi_{j-1}
    s2 = float(np.sum(c ** 2))
    s4 = float(np.sum(c ** 4))
    if not np.isfinite(gamma_delta) or s2 < 1e-12 or s4 < 1e-12:
        return 0.0
    return float(gamma_delta) * s2 ** 2 / s4


def _kurtosis_to_nu(gamma: float) -> float | None:
    """Student-t degrees of freedom whose excess kurtosis is ``gamma`` (6/(nu-4)).

    Returns None when the innovation is close enough to Gaussian that mixing is not
    worth the extra variance. nu is floored just above 4 because the fourth moment
    does not exist below it, and capped so a noisy kurtosis estimate cannot flip the
    generator into near-Gaussian behaviour by accident.
    """
    if not np.isfinite(gamma) or gamma < 0.05:
        return None
    return float(min(60.0, max(4.05, 4.0 + 6.0 / gamma)))


def _fit_var1(m: np.ndarray, idx: np.ndarray) -> dict:
    """VAR(1) by OLS on strictly consecutive day pairs.

    ``idx[j]`` marks days whose predecessor in the sample really is the previous
    calendar day. Fitting on the raw concatenation would pair 31 Jan of one year
    with 1 Jan of the next (a 335-day gap) once per year of record.
    """
    x = np.column_stack([np.ones(len(idx)), m[idx - 1]])
    y = m[idx]
    good = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x, y = x[good], y[good]
    if len(y) < 50:
        k = m.shape[1]
        return {"c": np.zeros(k).tolist(), "A": np.zeros((k, k)).tolist(),
                "sigma": np.eye(k).tolist(), "n_pairs": int(len(y)),
                "init_mean": np.nan_to_num(np.nanmean(m, 0)).tolist(),
                "init_cov": np.eye(k).tolist(), "max_eig": 0.0}
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    intercept = beta[0]
    a_mat = beta[1:].T
    resid = y - x @ beta
    sigma = resid.T @ resid / max(len(y) - x.shape[1], 1)
    eig = float(np.abs(np.linalg.eigvals(a_mat)).max())
    if eig >= 0.999:  # keep the sampler stable; shrinking is the lesser evil
        a_mat = a_mat * (0.999 / eig)
        eig = 0.999
    sigma = np.nan_to_num(sigma) + 1e-12 * np.eye(m.shape[1])
    return {"c": intercept.tolist(), "A": a_mat.tolist(), "sigma": sigma.tolist(),
            "n_pairs": int(len(y)), "max_eig": round(eig, 4),
            "init_mean": np.nan_to_num(np.nanmean(m, 0)).tolist(),
            "init_cov": np.nan_to_num(np.cov(m[~np.isnan(m).any(axis=1)].T)).tolist()}


def _innovation_corr(target: np.ndarray, polys: dict[str, tuple]) -> np.ndarray:
    """Innovation correlation R_e whose ARMA-filtered output has correlation R.

    For linear Gaussian filters, corr(u_t, v_t) = [R_e]_uv * <psi_u,psi_v> /
    (|psi_u||psi_v|), so divide the target elementwise by that cosine similarity.
    """
    from statsmodels.tsa.arima_process import arma2ma

    names = list(VARS)
    psi = {v: arma2ma(np.asarray(polys[v][0], float), np.asarray(polys[v][1], float),
                      lags=400) for v in names}
    n = len(names)
    cos = np.eye(n)
    for i in range(n):
        for k in range(n):
            a, b = psi[names[i]], psi[names[k]]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos[i, k] = float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 1.0
    r_e = target / np.clip(np.abs(cos), 1e-3, None)
    r_e = (r_e + r_e.T) / 2.0
    np.fill_diagonal(r_e, 1.0)
    r_e = np.clip(r_e, -0.999, 0.999)
    np.fill_diagonal(r_e, 1.0)
    w, vec = np.linalg.eigh(r_e)
    if w.min() <= 1e-8:  # nearest PSD with unit diagonal
        r_e = vec @ np.diag(np.clip(w, 1e-6, None)) @ vec.T
        d = np.sqrt(np.diag(r_e))
        r_e = r_e / np.outer(d, d)
    return r_e


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit(site: str, fit_years: tuple[int, int] = DEFAULT_FIT_YEARS) -> dict:
    """Fit the 12 monthly model groups and write output/synth/<site>_model.json."""
    raw = load_years(site, fit_years)
    data = clearness(raw, site)
    months = data.index.month.to_numpy()
    hours = data.index.hour.to_numpy()
    dnum = data.index.normalize().to_numpy().astype("datetime64[D]").astype(np.int64)

    months_out: dict[str, dict] = {}
    for month in range(1, 13):
        sel = months == month
        if sel.sum() < 200:
            continue
        sub = data.loc[sel]
        m_hours = hours[sel]
        m_dnum = dnum[sel]
        uniq, inv = np.unique(m_dnum, return_inverse=True)
        n_days = len(uniq)

        # --- (month, hour) climatology, used to standardise before anything else.
        clim: dict[str, dict] = {}
        anom: dict[str, np.ndarray] = {}
        scores: dict[str, np.ndarray] = {}
        for var in VARS:
            values = sub[var].to_numpy(float)
            mean = np.full(24, np.nan)
            std = np.full(24, np.nan)
            for h in range(24):
                v = values[m_hours == h]
                v = v[np.isfinite(v)]
                if v.size >= 20:
                    mean[h], std[h] = v.mean(), v.std(ddof=1)
            std = np.where(np.isfinite(std) & (std > 1e-9), std, np.nan)
            a = (values - mean[m_hours]) / std[m_hours]
            clim[var] = {"mean": np.nan_to_num(mean).round(6).tolist(),
                         "std": np.nan_to_num(std, nan=1.0).round(6).tolist()}
            anom[var] = a
            scores[var] = normal_scores(a)

        # --- day level: daily mean normal score per variable, then VAR(1).
        daily = np.full((n_days, len(VARS)), np.nan)
        for k, var in enumerate(VARS):
            z = scores[var]
            ok = np.isfinite(z)
            count = np.bincount(inv[ok], minlength=n_days).astype(float)
            total = np.bincount(inv[ok], weights=z[ok], minlength=n_days)
            daily[:, k] = np.where(count > 0, total / np.maximum(count, 1.0), np.nan)
        consecutive = np.zeros(n_days, bool)
        consecutive[1:] = (uniq[1:] - uniq[:-1]) == 1
        var1 = _fit_var1(daily, np.where(consecutive)[0])

        # --- hour level: deviation from the day's own mean, ARMA per variable.
        resid_sd: dict[str, float] = {}
        arma: dict[str, dict] = {}
        same_day = np.zeros(len(m_dnum), bool)
        same_day[1:] = m_dnum[1:] == m_dnum[:-1]
        for k, var in enumerate(VARS):
            r = scores[var] - daily[inv, k]
            r[~np.isfinite(scores[var])] = np.nan
            sd = float(np.nanstd(r)) if np.isfinite(r).sum() > 50 else 1.0
            if sd < 1e-9:
                sd = 1.0
            resid_sd[var] = round(sd, 6)
            picked = fit_arima(r / sd)
            picked["acf1_model"] = round(_acf1_model(picked["ar"], picked["ma"]), 4)
            picked["acf1_real"] = round(_lag1(r), 4)
            # Within a day the day-level state is constant, so the increment of the
            # score IS the increment of the residual -- measure its kurtosis there.
            dz = _within_day_diff(scores[var], same_day)
            gamma = float(kurtosis(dz)) if dz.size > 200 else float("nan")
            picked["inc_kurtosis"] = None if not np.isfinite(gamma) else round(gamma, 4)
            nu = _kurtosis_to_nu(_innovation_kurtosis(gamma, picked["ar"], picked["ma"]))
            picked["nu"] = None if nu is None else round(nu, 4)
            arma[var] = picked

        # One mixing variable per time step serves all four variables (see the
        # multivariate-t note in sample_days), so the month carries a single nu. The
        # four estimates sit in a narrow band (typically 5-8) and the median keeps one
        # badly-behaved variable from dragging the whole month.
        nus = [arma[v]["nu"] for v in VARS if arma[v]["nu"] is not None]
        innov_nu = round(float(np.median(nus)), 4) if nus else None

        # --- cross-variable copula on the daylight rows where all four are defined.
        cols = np.column_stack([scores[v] for v in VARS])
        use = np.isfinite(cols).all(axis=1)
        if use.sum() < 100:
            continue
        corr = np.corrcoef(cols[use].T)
        corr = np.clip(np.nan_to_num(corr, nan=0.0), -0.999, 0.999)
        np.fill_diagonal(corr, 1.0)
        innov = _innovation_corr(corr, {v: (arma[v]["ar"], arma[v]["ma"]) for v in VARS})

        months_out[str(month)] = {
            "n_years": int(sub.index.year.nunique()),
            "n_days": int(n_days),
            "n_hours": int(sel.sum()),
            "clim": clim,
            "quantiles": {v: quantile_grid(anom[v]) for v in VARS},
            "var1": var1,
            "resid_sd": resid_sd,
            "arma": arma,
            "innov_nu": innov_nu,
            "copula_corr": corr.round(6).tolist(),
            "innov_corr": innov.round(6).tolist(),
            "copula_n": int(use.sum()),
            "kt_lag1_real": round(_lag1(scores["kt"]), 4),
            "day_lag1_real": {v: round(_lag1(daily[:, k]), 4) for k, v in enumerate(VARS)},
        }
        kt = arma["kt"]
        print(f"  month {month:02d} : kt arima{kt['order']} acf1 real {kt['acf1_real']:+.3f} "
              f"model {kt['acf1_model']:+.3f} | VAR1 eig {var1['max_eig']:.3f} "
              f"pairs {var1['n_pairs']} | day lag1 kt {_lag1(daily[:, 0]):+.3f} "
              f"T {_lag1(daily[:, 1]):+.3f} W {_lag1(daily[:, 2]):+.3f} | "
              f"inc kurt " + "/".join(f"{v[:4]} {arma[v]['inc_kurtosis']}" for v in VARS) +
              f" -> nu {innov_nu}")

    model = {
        "site": site,
        "label": SITES[site]["label"],
        "fit_years": list(fit_years),
        "kt_min_clear_wm2": KT_MIN_CLEAR,
        "kt_max": KT_MAX,
        "vars": list(VARS),
        "qprobs": QPROBS.round(6).tolist(),
        "months": months_out,
        "limitations": [
            "Empirical margins are stored on a 0.05%-99.95% quantile grid; values outside it are "
            "clipped, so the synthetic record cannot exceed the observed range.",
            "kt is undefined at night; its innovation there is drawn independently, so night-time "
            "temperature and wind keep their own ARMA dynamics and the day-level synoptic state, "
            "but no hour-scale coupling to a (nonexistent) clearness index.",
            "The copula correlation is constant within a month; the real kt-temperature dependence "
            "varies with hour of day (temperature lags irradiance) and this is not captured.",
            "Copula tail dependence is that of a single-month multivariate t; the real "
            "dependence is stronger in some season/month combinations than others and "
            "one nu per month cannot track that.",
            "Hourly innovations are heavy-tailed but identically distributed within a "
            "month: volatility clustering (a stormy hour tends to be followed by more "
            "variable hours) is not modelled, only the unconditional kurtosis.",
            "The day-level VAR(1) is linear and homoscedastic; blocking-regime persistence and "
            "asymmetric transitions (slow clearing, fast clouding) are not represented.",
            "ARMA order is picked by AIC on the normal scores, which optimises one-step "
            "density, not ramp statistics; intermittency metrics can still be off by tens "
            "of percent (see the validation table).",
            "One ERA5 grid cell feeds every node, so the synthetic weather is spatially uniform by "
            "construction.",
        ],
    }
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    path = SYNTH_DIR / f"{site}_model.json"
    path.write_text(json.dumps(model, indent=1), encoding="utf-8")
    print(f"model             : {path}")
    return model


def load_model(site: str) -> dict:
    path = SYNTH_DIR / f"{site}_model.json"
    if not path.exists():
        raise SystemExit(f"no fitted model at {path}; run --fit --site {site} first")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_days(site: str, model: dict, dates: list[pd.Timestamp],
                seed: int = SYNTH_SEED) -> pd.DataFrame:
    """Generate synthetic hourly weather for the given calendar dates.

    Returned frame has the same columns and (tz-naive wall clock) index convention as
    the ERA5 cache, so ``ieee13_era5.py`` consumes it unchanged.

    Everything is generated for the whole span at once rather than day by day: the
    hourly ARMA innovations are a *continuous* stream across day boundaries (the
    real residual process does not restart at midnight), and the day-level VAR(1)
    state is a single 4-vector walked forward through the whole span, across month
    boundaries too.
    """
    rng = np.random.default_rng(seed)
    dates = [d for d in dates if str(int(d.month)) in model["months"]]
    if not dates:
        raise SystemExit("no synthetic days produced (all dates landed outside fitted months)")

    # One clearsky + ERBS call for the whole span: pvlib per-day calls dominate the
    # runtime when generating thousands of days (validating 10 years = 3652 days).
    index = pd.DatetimeIndex(np.concatenate([pd.date_range(d, periods=24, freq="h").values for d in dates]))
    sky = _clearsky(site, index)
    # kt is a ratio against the hour-average clearsky (see _clearsky), so the
    # generated GHI is kt * ghi_clear_avg and the daylight mask follows the same
    # quantity. ghi_clear (instantaneous) is still used for the ramp/drop-event
    # threshold, which is defined on the day's peak clearsky.
    ghi_clear = np.asarray(sky["ghi_clear"], float).reshape(len(dates), 24)
    ghi_clear_avg = np.asarray(sky["ghi_clear_avg"], float).reshape(len(dates), 24)
    clear_ref = ghi_clear_avg
    zenith = np.asarray(sky["zenith"], float).reshape(len(dates), 24)
    day = clear_ref > KT_MIN_CLEAR
    span = len(dates) * 24

    # --- day level: one VAR(1) walk, state carried across month boundaries.
    # NOTE: no post-hoc variance correction is applied to the day-level walk. An OLS
    # VAR(1) already reproduces the empirical covariance of the daily means exactly
    # (checked: solve_discrete_lyapunov(A, Sigma) == np.cov(daily) to 3 decimals for
    # every month and site), so the realised covariance of a synthetic block deviates
    # from the target only by the sampling variability of a persistent process -- at
    # phi ~ 0.85 a 310-day month has an effective sample size near 25, hence the
    # observed 0.86x-1.26x spread. Rescaling the realised covariance to the target
    # was tried and rejected: it removes real month-to-month climate variability (the
    # opposite of what this generator is for) and forces the synthetic sample onto the
    # fit period, which makes the validation metrics look better than they are.
    day_level = np.zeros((len(dates), len(VARS)))
    state: np.ndarray | None = None
    for i, date in enumerate(dates):
        grp = model["months"][str(int(date.month))]
        v = grp["var1"]
        c = np.asarray(v["c"], float)
        a_mat = np.asarray(v["A"], float)
        try:
            chol = np.linalg.cholesky(np.asarray(v["sigma"], float))
        except np.linalg.LinAlgError:
            chol = np.eye(len(c))
        if state is None:
            cov0 = np.nan_to_num(np.asarray(v["init_cov"], float)) + 1e-9 * np.eye(len(c))
            state = np.asarray(v["init_mean"], float) + np.linalg.cholesky(cov0) @ rng.standard_normal(len(c))
        state = c + a_mat @ state + chol @ rng.standard_normal(len(c))
        day_level[i] = state

    # --- hour level: continuous correlated innovations, filtered per variable.
    #
    # The innovations are multivariate Student-t, not Gaussian. Measured on the real
    # normal scores, the within-day increment has excess kurtosis of +1.4 to +12
    # depending on site, month and variable, against 0 for a Gaussian: real weather
    # makes fewer medium-size steps and several times more extreme ones (P(|dz| > 3)
    # = 0.011 against 0.003 in Atlanta in January). Drop-event and ramp thresholds sit
    # right in the shoulders of that distribution, so a Gaussian drive puts roughly
    # 2.7x too many hours over them.
    #
    # One chi-square mixing variable per time step is shared by all four variables,
    # which is what makes this a proper multivariate t with scale R rather than four
    # independent heavy-tailed series: dividing by sqrt(W/nu) leaves Var = R * nu /
    # (nu-2), so the extra sqrt((nu-2)/nu) puts the correlation back on R exactly. It
    # also gives the copula tail dependence the Gaussian version lacked. The cost is
    # a single nu per month (the median across variables) instead of one each.
    innov_stream = np.zeros((span, len(VARS)))
    for i, date in enumerate(dates):
        grp = model["months"][str(int(date.month))]
        r_e = np.asarray(grp["innov_corr"], float)
        try:
            chol = np.linalg.cholesky(r_e)
        except np.linalg.LinAlgError:
            w, vec = np.linalg.eigh(r_e)
            chol = vec @ np.diag(np.sqrt(np.clip(w, 1e-8, None))) @ vec.T
        block = slice(i * 24, (i + 1) * 24)
        gauss = rng.standard_normal((24, len(VARS))) @ chol.T
        nu = float(grp.get("innov_nu") or 0.0)
        if nu > 4.0:
            mix = rng.chisquare(nu, size=24)
            gauss = gauss / np.sqrt(mix / nu)[:, None] / np.sqrt(nu / (nu - 2.0))
        innov_stream[block] = gauss

    def grp_scale(month_key: str, var: str) -> float:
        return float(model["months"][month_key]["resid_sd"][var])

    scores = np.zeros((len(dates), 24, len(VARS)))
    for k, var in enumerate(VARS):
        # Per-month ARMA coefficients: filter the whole span with a piecewise-constant
        # model by running each day on its own month's filter. The transient is
        # absorbed with a burn-in prefix drawn from the same stream.
        x = np.zeros(span)
        burnin = 200
        for i, date in enumerate(dates):
            grp = model["months"][str(int(date.month))]
            a_ar = np.asarray(grp["arma"][var]["ar"], float)
            a_ma = np.asarray(grp["arma"][var]["ma"], float)
            e = innov_stream[i * 24:(i + 1) * 24, k]
            pre = rng.standard_normal(burnin)
            y = lfilter(a_ma, a_ar, np.concatenate([pre, e]))
            x[i * 24:(i + 1) * 24] = y[burnin:]
        x = x.reshape(len(dates), 24)
        if var == "kt":
            x = np.where(day, x, np.nan)
        # Scale to the fitted residual spread, per month. Deliberately NO per-day
        # re-centring: the ARMA was fitted on the residual that already had its day
        # mean removed, so its ACF is the post-removal one. Removing the day mean a
        # second time at generation pushes the within-day autocorrelation down again
        # (kt: 0.86 instead of the observed 0.92) and inflates hourly ramps by ~25%.
        # The cost is that the hourly component keeps a small day-scale part of its
        # own, so the synthetic day-mean variance sits ~6% above the fitted one --
        # the lesser of the two errors, and it is visible in the validation table.
        for month_key in sorted({str(int(d.month)) for d in dates}):
            rows = [i for i, d in enumerate(dates) if str(int(d.month)) == month_key]
            vals = x[rows][day[rows]] if var == "kt" else x[rows]
            cur = float(np.nanstd(vals))
            scale = grp_scale(month_key, var) / cur if cur > 1e-9 else 0.0
            x[rows] = x[rows] * scale
        scores[:, :, k] = day_level[:, k][:, None] + np.nan_to_num(x)

    kt_out = np.zeros((len(dates), 24))
    temp_out = np.zeros_like(kt_out)
    w100_out = np.zeros_like(kt_out)
    w10_out = np.zeros_like(kt_out)
    ghi_out = np.zeros_like(kt_out)

    for i, date in enumerate(dates):
        grp = model["months"][str(int(date.month))]
        clim = grp["clim"]
        quants = grp["quantiles"]
        dmask = day[i]
        for k, var in enumerate(VARS):
            a = quantile_apply(quants[var], norm.cdf(scores[i, :, k]))
            mean_h = np.array(clim[var]["mean"], float)
            std_h = np.array(clim[var]["std"], float)
            value = mean_h + std_h * a
            if var == "kt":
                kt_out[i] = np.where(dmask, np.clip(value, 0.0, KT_MAX), 0.0)
            elif var == "temp_air":
                temp_out[i] = np.clip(value, -60.0, 60.0)
            elif var == "wind_speed_100m":
                w100_out[i] = np.clip(value, 0.0, 60.0)
            else:
                w10_out[i] = np.clip(value, 0.0, 50.0)
        ghi_out[i] = np.where(dmask, kt_out[i] * clear_ref[i], 0.0)

    # --- DNI/DHI: ERBS decomposition keeps the three components mutually consistent.
    erbs = pvlib.irradiance.erbs(ghi_out.ravel(), zenith.ravel(), index.dayofyear.to_numpy())
    dni = np.nan_to_num(np.asarray(erbs["dni"], float), nan=0.0).reshape(len(dates), 24)
    dhi = np.nan_to_num(np.asarray(erbs["dhi"], float), nan=0.0).reshape(len(dates), 24)
    dni = np.where(day, np.clip(dni, 0.0, None), 0.0)
    dhi = np.where(day, np.clip(dhi, 0.0, None), 0.0)

    return pd.DataFrame(
        {"ghi": np.clip(ghi_out, 0.0, None).ravel(), "dni": dni.ravel(), "dhi": dhi.ravel(),
         "temp_air": temp_out.ravel(), "wind_speed": w10_out.ravel(),
         "wind_speed_100m": w100_out.ravel()},
        index=index,
    ).sort_index()


def sample_weather(site: str, start: pd.Timestamp, days: int,
                   seed: int = SYNTH_SEED, model: dict | None = None) -> pd.DataFrame:
    """Synthetic weather covering ``days`` consecutive days from ``start``.

    No DST skipping here: the ERA5 cache is a fixed-offset clock (see SITES), so
    every calendar day really does have 24 rows and every day is usable.
    """
    model = model or load_model(site)
    dates = [start + pd.Timedelta(days=i) for i in range(days)]
    return sample_days(site, model, dates, seed=seed)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _daily_energy(frame: pd.DataFrame) -> pd.Series:
    return frame["ghi"].groupby(frame.index.date).sum() / 1000.0


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "q05": float(np.quantile(x, 0.05)), "q25": float(np.quantile(x, 0.25)),
            "q50": float(np.quantile(x, 0.50)), "q75": float(np.quantile(x, 0.75)),
            "q95": float(np.quantile(x, 0.95)), "min": float(x.min()), "max": float(x.max())}


def validate(site: str, val_years: tuple[int, int] = DEFAULT_VAL_YEARS,
             seed: int = SYNTH_SEED, synth_multiple: float = 3.0) -> pd.DataFrame:
    """Compare synthetic weather against held-out real years.

    The synthetic side is deliberately ``synth_multiple`` times larger than the real
    one: generating more days than were observed is the entire point of the model, and
    it also takes sampling noise off the synthetic side of every statistic, so what is
    left in the Wasserstein distance is model error rather than Monte-Carlo error.
    """
    model = load_model(site)
    real = clearness(load_years(site, val_years), site)
    # The synthetic set reproduces the FIT period, so every gap against a held-out
    # period is a mix of model error and genuine climate drift (1990-2014 is colder
    # than 2015-2024, and ERA5 100 m wind is not homogeneous across the record).
    # Carrying the fit-period statistics along is what separates the two.
    fit_period = clearness(load_years(site, tuple(model["fit_years"])), site)

    n_real = len({d for d in real.index.date})
    synth_dates = [pd.Timestamp(d) for d in
                   pd.date_range("1950-01-01", periods=int(round(n_real * synth_multiple)), freq="D")]
    synth = clearness(sample_days(site, model, synth_dates, seed=seed), site)

    def series(frame: pd.DataFrame, col: str) -> np.ndarray:
        return frame[col].to_numpy(float)

    metrics = []

    def add(name, unit, real_v, synth_v, fit_v=None):
        rv = np.asarray(real_v, float); sv = np.asarray(synth_v, float)
        rv = rv[np.isfinite(rv)]; sv = sv[np.isfinite(sv)]
        if rv.size < 10 or sv.size < 10:
            return
        rs, ss = _stats(rv), _stats(sv)
        fs = _stats(np.asarray(fit_v, float)) if fit_v is not None else None
        w = float(wasserstein_distance(rv, sv))
        qdev = {f"q{p:02d}_dev_pct": round((ss[f"q{p:02d}"] - rs[f"q{p:02d}"]) /
                                           (abs(rs[f"q{p:02d}"]) + 1e-12) * 100, 2)
                for p in (5, 25, 50, 75, 95)}
        metrics.append({"metric": name, "unit": unit, "n_real": int(rv.size), "n_synth": int(sv.size),
                        "real_mean": round(rs["mean"], 4), "synth_mean": round(ss["mean"], 4),
                        "real_std": round(rs["std"], 4), "synth_std": round(ss["std"], 4),
                        "real_q05": round(rs["q05"], 4), "synth_q05": round(ss["q05"], 4),
                        "real_q95": round(rs["q95"], 4), "synth_q95": round(ss["q95"], 4),
                        "fit_mean": round(fs["mean"], 4) if fs else None,
                        "fit_std": round(fs["std"], 4) if fs else None,
                        "wasserstein": round(w, 4),
                        "wasserstein_pct_of_mean": round(w / (abs(rs["mean"]) + 1e-12) * 100, 2),
                        **qdev})

    def ghi_daylight(frame: pd.DataFrame) -> np.ndarray:
        return series(frame, "ghi")[np.isfinite(series(frame, "kt"))]

    re_day = _daily_energy(real); sy_day = _daily_energy(synth)
    add("daily GHI energy", "kWh/m2/day", re_day.to_numpy(), sy_day.to_numpy(),
        _daily_energy(fit_period).to_numpy())
    add("hourly GHI (daylight)", "W/m2", ghi_daylight(real), ghi_daylight(synth),
        ghi_daylight(fit_period))
    add("hourly kt", "-", series(real, "kt"), series(synth, "kt"), series(fit_period, "kt"))
    add("hourly temp_air", "degC", series(real, "temp_air"), series(synth, "temp_air"),
        series(fit_period, "temp_air"))
    add("hourly wind 100 m", "m/s", series(real, "wind_speed_100m"), series(synth, "wind_speed_100m"),
        series(fit_period, "wind_speed_100m"))
    add("daily mean temp", "degC",
        real["temp_air"].groupby(real.index.date).mean().to_numpy(),
        synth["temp_air"].groupby(synth.index.date).mean().to_numpy(),
        fit_period["temp_air"].groupby(fit_period.index.date).mean().to_numpy())
    add("daily mean wind 100 m", "m/s",
        real["wind_speed_100m"].groupby(real.index.date).mean().to_numpy(),
        synth["wind_speed_100m"].groupby(synth.index.date).mean().to_numpy(),
        fit_period["wind_speed_100m"].groupby(fit_period.index.date).mean().to_numpy())

    # --- dependence / dynamics metrics, which the marginals cannot see.
    def kt_daily_mean(frame: pd.DataFrame) -> np.ndarray:
        s = frame["kt"].groupby(frame.index.date).mean()
        return s.to_numpy(float)

    def kt_hourly_lag1(frame: pd.DataFrame) -> float:
        s = frame["kt"].to_numpy(float)
        # index.date must be materialised once: calling it inside the comprehension
        # rebuilt an 87k-element object array 87k times, which is what made the
        # 10-year validation run for >40 minutes.
        dates = frame.index.normalize().to_numpy()
        same = dates[:-1] == dates[1:]
        ok = np.isfinite(s[:-1]) & np.isfinite(s[1:]) & same
        a, b = s[:-1][ok], s[1:][ok]
        return float(np.corrcoef(a, b)[0, 1]) if ok.sum() > 50 else float("nan")

    def ramp_p95(frame: pd.DataFrame) -> np.ndarray:
        g = frame["ghi"].groupby(frame.index.date)
        return np.array([np.nanpercentile(np.abs(np.diff(v.to_numpy(float))), 95) for _, v in g])

    def drop_events(frame: pd.DataFrame, frac: float = 0.30) -> np.ndarray:
        # Threshold on the day's peak hour-average clearsky: that is the quantity the
        # GHI series itself is normalised by, and the instantaneous peak is ~8% higher,
        # which is enough to move the event count by a third.
        col = "ghi_clear_avg" if "ghi_clear_avg" in frame.columns else "ghi_clear"
        clear = frame[col].groupby(frame.index.date)
        ghi = frame["ghi"].groupby(frame.index.date)
        out = []
        for (_, c), (_, g) in zip(clear, ghi):
            c = c.to_numpy(float); g = g.to_numpy(float)
            peak = np.nanmax(c)
            if peak <= KT_MIN_CLEAR:
                out.append(0.0); continue
            out.append(float((np.diff(g) < -frac * peak).sum()))
        return np.array(out)

    tl1_r, tl1_s = _lag1(kt_daily_mean(real)), _lag1(kt_daily_mean(synth))
    hl1_r, hl1_s = kt_hourly_lag1(real), kt_hourly_lag1(synth)
    add("daily kt lag-1 autocorr (5-y blocks)", "-",
        np.array([_lag1(v) for v in np.array_split(kt_daily_mean(real), 10)]),
        np.array([_lag1(v) for v in np.array_split(kt_daily_mean(synth), 10)]))
    add("daily GHI ramp P95", "W/m2/h", ramp_p95(real), ramp_p95(synth), ramp_p95(fit_period))
    add("GHI drop events > 30% clear / day", "count", drop_events(real), drop_events(synth),
        drop_events(fit_period))

    frame = pd.DataFrame(metrics)
    tl1_f, hl1_f = _lag1(kt_daily_mean(fit_period)), kt_hourly_lag1(fit_period)
    frame.loc[len(frame)] = {"metric": "kt lag-1 autocorr: DAILY mean", "unit": "-",
                             "real_mean": round(tl1_r, 4), "synth_mean": round(tl1_s, 4),
                             "fit_mean": round(tl1_f, 4),
                             "wasserstein": round(abs(tl1_s - tl1_r), 4)}
    frame.loc[len(frame)] = {"metric": "kt lag-1 autocorr: HOURLY within day", "unit": "-",
                             "real_mean": round(hl1_r, 4), "synth_mean": round(hl1_s, 4),
                             "fit_mean": round(hl1_f, 4),
                             "wasserstein": round(abs(hl1_s - hl1_r), 4)}

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SYNTH_DIR / f"{site}_validation.csv", index=False)
    print(f"\nvalidation        : {SYNTH_DIR / f'{site}_validation.csv'}")
    print(f"real days         : {n_real} ({val_years[0]}-{val_years[1]}), "
          f"synthetic days: {len(synth_dates)} ({synth_multiple:g}x)")
    print()
    print(frame[["metric", "unit", "fit_mean", "real_mean", "synth_mean",
                 "fit_std", "real_std", "synth_std",
                 "wasserstein", "wasserstein_pct_of_mean"]].to_string(index=False))
    return frame


# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ARIMA(kt) + Copula synthetic ERA5 weather.")
    parser.add_argument("--site", default="atlanta", choices=sorted(SITES))
    parser.add_argument("--fit", action="store_true", help="Fit the 12 monthly model groups.")
    parser.add_argument("--validate", action="store_true", help="Validate against held-out years.")
    parser.add_argument("--fit-years", type=int, nargs=2, default=list(DEFAULT_FIT_YEARS))
    parser.add_argument("--val-years", type=int, nargs=2, default=list(DEFAULT_VAL_YEARS))
    parser.add_argument("--seed", type=int, default=SYNTH_SEED)
    args = parser.parse_args(argv)

    if args.fit:
        print(f"fitting {args.site} on {args.fit_years[0]}-{args.fit_years[1]}")
        fit(args.site, tuple(args.fit_years))
    if args.validate:
        print(f"validating {args.site} on {args.val_years[0]}-{args.val_years[1]}")
        validate(args.site, tuple(args.val_years), seed=args.seed)
    if not args.fit and not args.validate:
        parser.print_help()


if __name__ == "__main__":
    _main()
