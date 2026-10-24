"""Linear advection-diffusion SPDEs on a physical, rectangular metre grid.

Implicit time stepping, upwind advection with zero-anomaly inflow, and zero-flux
diffusion. Joint fields have separate operators and spatial smoothers; their
Gaussian driving noises are coupled through a validated correlation matrix.
This is a statistical residual-field model, not an atmospheric flow solver.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_discrete_lyapunov
from scipy.sparse.linalg import splu


def implementation_fingerprint() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = ("src/weather_spde.py", "src/spde_case.py", "data/generate_data/ieee13_era5.py",
             "src/regional_weather.py", "src/regional_spde.py", "src/weather_io.py",
             "src/site_weather.py", "data/generate_data/station_spde.py",
             "data/generate_data/regional_spde.py")
    packages = {}
    for name in ("numpy", "scipy", "pandas", "pvlib", "dss-python"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
            "packages": packages}


@dataclass(frozen=True)
class RectGrid:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        for name in ("x", "y"):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
                raise ValueError("Each grid axis needs at least two finite coordinates.")
            delta = np.diff(a)
            if np.any(delta <= 0) or not np.allclose(delta, delta[0]):
                raise ValueError("Grid axes must be increasing and uniformly spaced.")
            a.flags.writeable = False
            object.__setattr__(self, name, a)

    @property
    def size(self) -> int:
        return len(self.x) * len(self.y)

    @property
    def xy(self) -> np.ndarray:
        xx, yy = np.meshgrid(self.x, self.y)
        return np.column_stack((xx.ravel(), yy.ravel()))

    def observation_matrix(self, xy: np.ndarray) -> sparse.csr_matrix:
        """Bilinear interpolation; reject extrapolation outside the domain."""
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("Station coordinates must have shape (stations, 2) and be finite.")
        if (np.any(xy[:, 0] < self.x[0]) or np.any(xy[:, 0] > self.x[-1])
                or np.any(xy[:, 1] < self.y[0]) or np.any(xy[:, 1] > self.y[-1])):
            raise ValueError("Station lies outside the SPDE grid.")
        ix = np.clip(np.searchsorted(self.x, xy[:, 0], side="right") - 1, 0, len(self.x) - 2)
        iy = np.clip(np.searchsorted(self.y, xy[:, 1], side="right") - 1, 0, len(self.y) - 2)
        wx = (xy[:, 0] - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        wy = (xy[:, 1] - self.y[iy]) / (self.y[iy + 1] - self.y[iy])
        columns = np.column_stack((iy * len(self.x) + ix, iy * len(self.x) + ix + 1,
                                   (iy + 1) * len(self.x) + ix, (iy + 1) * len(self.x) + ix + 1))
        weights = np.column_stack(((1-wx)*(1-wy), wx*(1-wy), (1-wx)*wy, wx*wy))
        return sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(len(xy)), 4), columns.ravel())),
                                 shape=(len(xy), self.size))


@dataclass(frozen=True)
class FieldParameters:
    name: str
    velocity_x_m_s: float = 0.0
    velocity_y_m_s: float = 0.0
    diffusivity_m2_s: float = 100.0
    decay_per_s: float = 1 / 1800
    noise_std_per_sqrt_s: float = 0.01
    noise_length_m: float = 1000.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Field name must not be empty.")
        numeric = {k: v for k, v in asdict(self).items() if k != "name"}
        if not all(np.isfinite(v) for v in numeric.values()):
            raise ValueError("Field parameters must be finite.")
        if min(self.diffusivity_m2_s, self.decay_per_s,
               self.noise_std_per_sqrt_s, self.noise_length_m) < 0:
            raise ValueError("Diffusion, damping, noise and length must be non-negative.")


def _axis_operator(n: int, spacing: float, velocity: float, diffusion: float) -> sparse.csc_matrix:
    # Neumann diffusion preserves constants, including at boundary cells.
    diag = np.full(n, -2.0)
    diag[[0, -1]] = -1.0
    lap = sparse.diags((np.ones(n-1), diag, np.ones(n-1)), (-1, 0, 1), format="csc") / spacing**2
    advection = sparse.diags(np.full(n, -abs(velocity) / spacing), format="csc")
    if velocity:
        advection += sparse.diags(np.full(n-1, abs(velocity) / spacing),
                                  -1 if velocity > 0 else 1, shape=(n, n), format="csc")
    return diffusion * lap + advection


def evolution_operator(grid: RectGrid, p: FieldParameters) -> sparse.csc_matrix:
    ax = _axis_operator(len(grid.x), grid.x[1]-grid.x[0], p.velocity_x_m_s, p.diffusivity_m2_s)
    ay = _axis_operator(len(grid.y), grid.y[1]-grid.y[0], p.velocity_y_m_s, p.diffusivity_m2_s)
    return (sparse.kron(sparse.eye(len(grid.y)), ax, format="csc")
            + sparse.kron(ay, sparse.eye(len(grid.x)), format="csc")
            - p.decay_per_s * sparse.eye(grid.size, format="csc"))


class JointWeatherSPDE:
    """Joint Gaussian latent fields. Sampling never forms a dense global covariance.

    Noise smoother S=(I-length²*Laplacian)^-1. It is normalized to unit average
    grid-node variance; sigma is thus a grid-RMS driving intensity. Normalization
    is exact up to 256 nodes, otherwise uses 64 deterministic Rademacher probes.
    Spatial resolution remains an explicit modeling choice, not an observation.
    """

    def __init__(self, grid: RectGrid, fields: Sequence[FieldParameters], dt_seconds: float,
                 driving_correlation: np.ndarray | None = None):
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive and finite.")
        self.grid, self.fields, self.dt_seconds = grid, tuple(fields), float(dt_seconds)
        if not self.fields or len({p.name for p in self.fields}) != len(self.fields):
            raise ValueError("Use at least one field, with unique names.")
        count = len(self.fields)
        correlation = np.eye(count) if driving_correlation is None else np.asarray(driving_correlation, float)
        if (correlation.shape != (count, count) or not np.isfinite(correlation).all()
                or not np.allclose(correlation, correlation.T, atol=1e-10)
                or not np.allclose(np.diag(correlation), 1.0, atol=1e-10)):
            raise ValueError("Driving correlation must be finite, symmetric, with unit diagonal.")
        values, vectors = np.linalg.eigh(correlation)
        if values.min() < -1e-10:
            raise ValueError("Driving correlation must be positive semidefinite.")
        self.driving_correlation = correlation.copy()
        self._mix = vectors @ np.diag(np.sqrt(np.maximum(values, 0)))
        eye = sparse.eye(grid.size, format="csc")
        self._steps, self._smoothers, self._norms = [], [], []
        lap = evolution_operator(grid, FieldParameters("laplacian", diffusivity_m2_s=1, decay_per_s=0,
                                                      noise_std_per_sqrt_s=0))
        smoother_cache: dict[float, tuple[object, float]] = {}
        for p in self.fields:
            self._steps.append(splu(eye - self.dt_seconds * evolution_operator(grid, p)))
            if p.noise_length_m not in smoother_cache:
                smooth = splu(eye - p.noise_length_m**2 * lap)
                if grid.size <= 256:
                    basis = smooth.solve(np.eye(grid.size))
                    norm = float(np.sqrt(np.sum(basis**2) / grid.size))
                else:
                    probes = np.random.default_rng(1907).choice([-1., 1.], size=(grid.size, 64))
                    norm = float(np.sqrt(np.mean(smooth.solve(probes)**2)))
                smoother_cache[p.noise_length_m] = (smooth, norm)
            smooth, norm = smoother_cache[p.noise_length_m]
            self._smoothers.append(smooth)
            self._norms.append(norm)

    def step(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Advance (..., fields, nodes); all fields share the same physical grid."""
        state = np.asarray(state, dtype=float)
        expected = (len(self.fields), self.grid.size)
        if state.shape[-2:] != expected or not np.isfinite(state).all():
            raise ValueError(f"State must be finite and end in shape {expected}.")
        flat = state.reshape((-1, *expected))
        independent = rng.standard_normal(flat.shape)
        coupled = np.einsum("ij,bjn->bin", self._mix, independent)
        out = np.empty_like(flat)
        for i, p in enumerate(self.fields):
            forcing = self._smoothers[i].solve(coupled[:, i, :].T).T / self._norms[i]
            rhs = flat[:, i, :] + np.sqrt(self.dt_seconds) * p.noise_std_per_sqrt_s * forcing
            out[:, i, :] = self._steps[i].solve(rhs.T).T
        return out.reshape(state.shape)

    def sample_sites(self, xy: np.ndarray, steps: int, rng: np.random.Generator,
                     *, burnin_steps: int = 0, initial_state: np.ndarray | None = None) -> np.ndarray:
        """Return (time, sites, fields), after one transition per recorded time."""
        if steps < 1 or burnin_steps < 0:
            raise ValueError("steps must be positive and burnin_steps non-negative.")
        h = self.grid.observation_matrix(xy)
        state = np.zeros((len(self.fields), self.grid.size)) if initial_state is None else np.array(initial_state, float)
        if state.shape != (len(self.fields), self.grid.size):
            raise ValueError("initial_state must have shape (fields, nodes).")
        for _ in range(burnin_steps):
            state = self.step(state, rng)
        samples = np.empty((steps, len(xy), len(self.fields)))
        for t in range(steps):
            state = self.step(state, rng)
            samples[t] = h @ state.T
        return samples

    def dense_single_field(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Small-grid inference only: transition F, noise factor G, stationary P."""
        if len(self.fields) != 1 or self.grid.size > 256:
            raise ValueError("Dense inference is limited to one field and at most 256 nodes.")
        f = self._steps[0].solve(np.eye(self.grid.size))
        g = (np.sqrt(self.dt_seconds) * self.fields[0].noise_std_per_sqrt_s
             * self._steps[0].solve(self._smoothers[0].solve(np.eye(self.grid.size))) / self._norms[0])
        if np.max(np.abs(np.linalg.eigvals(f))) >= 1 - 1e-12:
            raise ValueError("Stationary inference needs a strictly stable evolution operator.")
        p = solve_discrete_lyapunov(f, g @ g.T)
        return f, g, (p + p.T) / 2

    def dense_joint(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Small joint-grid inference; field-major states (field, then node)."""
        if self.grid.size*len(self.fields) > 256:
            raise ValueError("Dense joint inference is limited to 256 total states.")
        fs, gs = [], []
        eye = np.eye(self.grid.size)
        for i, field in enumerate(self.fields):
            fs.append(self._steps[i].solve(eye))
            gs.append(np.sqrt(self.dt_seconds)*field.noise_std_per_sqrt_s
                      *self._steps[i].solve(self._smoothers[i].solve(eye))/self._norms[i])
        f = block_diag(*fs)
        g = block_diag(*gs) @ np.kron(self._mix, eye)
        if np.max(np.abs(np.linalg.eigvals(f))) >= 1-1e-12:
            raise ValueError("Stationary inference needs strictly stable fields.")
        p = solve_discrete_lyapunov(f, g@g.T)
        return f, g, (p+p.T)/2


def kalman_update(mean: np.ndarray, cov: np.ndarray, row: np.ndarray, h: np.ndarray,
                  observation_std: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """One observation update, accepting missing values and per-channel errors."""
    std = np.broadcast_to(np.asarray(observation_std, float), (len(row),))
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("observation_std must be finite and positive for every channel.")
    valid = np.isfinite(row)
    if not valid.any():
        return mean.copy(), cov.copy(), 0.
    ht = h[valid]
    residual = row[valid]-ht@mean
    cross = cov@ht.T
    innovation = ht@cross+np.diag(std[valid]**2)
    factor = cho_factor(innovation, lower=True, check_finite=False)
    nll = .5*(len(residual)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()
              +residual@cho_solve(factor, residual, check_finite=False))
    gain = cho_solve(factor, cross.T, check_finite=False).T
    updated = cov-gain@cross.T
    return mean+gain@residual, (updated+updated.T)/2, float(nll)


def gaussian_filter(observations: np.ndarray, h: np.ndarray, f: np.ndarray,
                    q: np.ndarray, initial_cov: np.ndarray, observation_std: float | np.ndarray,
                    *, assimilate: np.ndarray | None = None,
                    store_history: bool = True) -> dict[str, np.ndarray | float]:
    """Kalman filter with missing data. Forecasts precede each time's update.

    assimilate=False makes that time a pure forecast, including all future times
    in a forecasting test. No smoothing or future observations are used.
    """
    y, h = np.asarray(observations, float), np.asarray(h, float)
    if y.ndim != 2 or h.ndim != 2 or y.shape[1] != h.shape[0]:
        raise ValueError("Observations and station operator have incompatible shapes.")
    std = np.broadcast_to(np.asarray(observation_std, float), (y.shape[1],))
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("observation_std must be positive.")
    if np.isinf(y).any() or not np.isfinite(h).all():
        raise ValueError("Observations may contain NaN, but not infinity; H must be finite.")
    use = np.ones(len(y), bool) if assimilate is None else np.asarray(assimilate, bool)
    if use.shape != (len(y),):
        raise ValueError("Assimilation mask has the wrong length.")
    mean = np.zeros(f.shape[0])
    cov = np.array(initial_cov, float, copy=True)
    means, covs, predictions, variances = [], [], [], []
    nll = 0.0
    for t, row in enumerate(y):
        if t:
            mean = f @ mean
            cov = f @ cov @ f.T + q
            cov = (cov + cov.T) / 2
        if store_history:
            predictions.append(h @ mean)
            variances.append(np.maximum(np.einsum("ij,jk,ik->i", h, cov, h), 0) + std**2)
        if use[t]:
            mean, cov, increment = kalman_update(mean, cov, row, h, std)
            nll += increment
        if store_history:
            means.append(mean.copy())
            covs.append(cov.copy())
    return {"nll": float(nll), "means": np.array(means), "covariances": np.array(covs),
            "predictions": np.array(predictions), "variances": np.array(variances)}


def ensemble_scores(observed: np.ndarray, samples: np.ndarray, xy: np.ndarray) -> dict:
    """Energy and distance-weighted variogram scores; lower is better.

    observed=(time, sites), samples=(members, time, sites). Scores target the
    supplied quantity and units. Add observation error to samples when scoring
    observed instruments rather than the latent process field.
    """
    observed, samples, xy = np.asarray(observed, float), np.asarray(samples, float), np.asarray(xy, float)
    if (observed.ndim != 2 or samples.ndim != 3 or samples.shape[1:] != observed.shape
            or xy.shape != (observed.shape[1], 2) or len(samples) < 2
            or not np.isfinite(samples).all() or not np.isfinite(xy).all()):
        raise ValueError("Use finite samples (>=2 members) and coordinates matching (time, sites) observations.")
    energy, variogram = [], []
    for t, row in enumerate(observed):
        valid = np.isfinite(row)
        if valid.sum() < 2:
            continue
        x, truth = samples[:, t, valid], row[valid]
        # Consecutive independent ensemble members estimate the second expectation.
        energy.append(float(np.linalg.norm(x-truth, axis=1).mean()
                            -.5*np.linalg.norm(x-np.roll(x, 1, axis=0), axis=1).mean()))
        i, j = np.triu_indices(len(truth), 1)
        distance = np.linalg.norm(xy[valid][i]-xy[valid][j], axis=1)
        weights = 1/np.maximum(distance, 500.)
        differences = np.sqrt(np.abs(truth[i]-truth[j]))-np.sqrt(np.abs(x[:, i]-x[:, j])).mean(axis=0)
        variogram.append(float(np.sum(weights*differences**2)/weights.sum()))
    if not energy:
        raise ValueError("At least two finite observed sites per scoring time are needed.")
    return {"energy_score": float(np.mean(energy)), "variogram_score_p_half": float(np.mean(variogram)),
            "scored_times": len(energy), "members": len(samples), "lower_is_better": True,
            "energy_second_term": "cyclic adjacent member pairs", "variogram_weights": "1/max(distance_m,500), normalized"}
