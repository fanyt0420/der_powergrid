"""History-conditioned neural coefficients for an implicit joint weather SPDE.

The network predicts physical coefficients; it never replaces the spatial PDE
step with a direct weather predictor. Training uses resolved ERA5 residuals as
state proxies (not a neural Kalman likelihood). Held-cell outcomes are excluded
from both training inputs and targets. This approximation is explicit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as TF

from src.weather_spde import RectGrid, FieldParameters, evolution_operator


def calendar_features(times):
    times = pd.DatetimeIndex(times)
    day = (times.dayofyear.to_numpy()-1+(times.hour.to_numpy()+.5)/24)/365.2425
    hour = (times.hour.to_numpy()+times.minute.to_numpy()/60)/24
    a = [np.ones(len(times))]
    for frequency in (1, 2, 3):
        a.extend([np.sin(2*np.pi*frequency*day), np.cos(2*np.pi*frequency*day)])
    for frequency in (1, 2):
        a.extend([np.sin(2*np.pi*frequency*hour), np.cos(2*np.pi*frequency*hour)])
    return np.column_stack(a).astype(np.float32)


def design_matrix(times, sites):
    calendar = calendar_features(times)
    xy = sites[["x_m", "y_m"]].to_numpy(np.float32)/100000
    design = np.broadcast_to(calendar[:, None], (len(times), len(sites), calendar.shape[1]))
    spatial = np.broadcast_to(xy[None], (len(times), len(sites), 2))
    cross = calendar[:, None, 1:3, None]*xy[None, :, None, :]
    return np.concatenate([design, spatial, cross.reshape(len(times), len(sites), 4)], axis=-1)


class MultiYearWeather:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory/"manifest.json").read_text())
        split = {key: set(self.manifest["split"][key]) for key in ("train", "validation", "test")}
        if any(not years for years in split.values()) or split["train"] & split["validation"] or split["train"] & split["test"] or split["validation"] & split["test"]:
            raise ValueError("Training, validation and test years must be nonempty and disjoint.")
        if not (split["train"] | split["validation"] | split["test"]).issubset({int(year) for year in self.manifest["years"]}):
            raise ValueError("Finish downloading all split years before training.")
        self.sites = pd.read_csv(self.directory/"sites.csv")
        self.held = np.array(self.manifest["held_sites"])
        self.train_sites = np.setdiff1d(np.arange(len(self.sites)), self.held)
        values = {key: [] for key in ("raw", "clear", "ghi", "times")}
        for year in sorted(self.manifest["years"]):
            entry = self.manifest["years"][year]
            path = self.directory/entry["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Multi-year source checksum mismatch.")
            with np.load(path) as source:
                for key in values:
                    values[key].append(source[key].copy())
        self.raw, self.clear, self.ghi = [np.concatenate(values[key]) for key in ("raw", "clear", "ghi")]
        self.times = pd.to_datetime(np.concatenate(values["times"]), utc=True)
        if not np.all(np.diff(self.times.asi8) == 3600*10**9):
            raise ValueError("Multi-year data must be uninterrupted hourly intervals.")
        self.calendar = calendar_features(self.times)
        self.design = design_matrix(self.times, self.sites)
        train_time = self.times.year.isin(self.manifest["split"]["train"])
        x = self.design[train_time][:, self.train_sites].reshape(-1, self.design.shape[-1])
        self.coefficients, scales = [], []
        for field in range(4):
            y = self.raw[train_time, field][:, self.train_sites].reshape(-1)
            good = np.isfinite(y)
            coef = np.linalg.lstsq(x[good].astype(float), y[good].astype(float), rcond=None)[0]
            self.coefficients.append(coef)
            scales.append(np.std(y[good]-x[good]@coef))
        self.coefficients = np.array(self.coefficients, np.float32)
        self.scale = np.array(scales, np.float32)
        if not np.isfinite(self.scale).all() or (self.scale <= 1e-6).any():
            raise ValueError("Training residual scales must be finite and nonzero.")
        self.background = np.einsum("tnk,fk->tfn", self.design, self.coefficients)
        self.z = (self.raw-self.background)/self.scale[None, :, None]
        self.valid = np.isfinite(self.z)
        self.z = np.nan_to_num(self.z).astype(np.float32)
        # Fixed observation map: never read held-cell histories, even at test time.
        xy = self.sites[["x_m", "y_m"]].to_numpy()
        self.projection = np.eye(len(xy), dtype=np.float32)
        for cell in self.held:
            distance = np.linalg.norm(xy[self.train_sites]-xy[cell], axis=1)
            nearest = np.argsort(distance)[:4]
            w = 1/np.maximum(distance[nearest], 1.)**2
            self.projection[cell] = 0.
            self.projection[cell, self.train_sites[nearest]] = w/w.sum()
        self.input_z = np.einsum("tfc,nc->tfn", self.z, self.projection)
        self.input_z[:, 0] = np.where(self.clear >= 50, self.input_z[:, 0], 0.)
        self.grid = RectGrid(np.sort(self.sites.x_m.unique()), np.sort(self.sites.y_m.unique()))
        # Avoid retaining a large calendar design after background preparation.
        del self.design

    def origins(self, split, *, history=24, horizon=24, stride=1):
        years = self.manifest["split"][split]
        candidates = np.arange(history-1, len(self.times)-horizon, stride)
        first = self.times[candidates+1].year.isin(years)
        last = self.times[candidates+horizon].year.isin(years)
        return candidates[first & last]

    def batch(self, origins, history, horizon, device):
        origins = np.asarray(origins, int)
        history_index = origins[:, None]+np.arange(-history+1, 1)
        future = origins[:, None]+np.arange(1, horizon+1)
        previous = self.valid[origins].copy()
        mask = self.valid[future].copy()
        mask[:, :, :, self.held] = False
        # Solar sunrise is not a known one-step latent transition from nighttime.
        mask[:, 0, 0] &= previous[:, 0]
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        return {"history": tensor(self.input_z[history_index]), "target": tensor(self.z[future]),
                "mask": tensor(mask), "calendar": tensor(self.calendar[future]),
                "daylight": tensor(self.clear[future] >= 50), "indices": future}


class CoefficientSPDE(nn.Module):
    """A differentiable implicit upwind/diffusion SPDE with shared neural coefficients.

    Each field has vx, vy [m/hour], diffusivity [m2/hour], decay [1/hour],
    driving intensity [latent/sqrt(hour)], noise length [m], equilibrium offset.
    A separate lower-triangular factor learns cross-field driving correlation.
    Bounds plus implicit stepping retain dissipative operators.
    """
    def __init__(self, grid, projection, *, history=24, width=512, depth=3, neural=True):
        super().__init__()
        self.history, self.neural = history, neural
        self.size = grid.size
        for name, matrix in (
            ("lap", evolution_operator(grid, FieldParameters("lap", diffusivity_m2_s=1, decay_per_s=0))),
            ("axp", evolution_operator(grid, FieldParameters("axp", velocity_x_m_s=1, diffusivity_m2_s=0, decay_per_s=0))),
            ("axn", evolution_operator(grid, FieldParameters("axn", velocity_x_m_s=-1, diffusivity_m2_s=0, decay_per_s=0))),
            ("ayp", evolution_operator(grid, FieldParameters("ayp", velocity_y_m_s=1, diffusivity_m2_s=0, decay_per_s=0))),
            ("ayn", evolution_operator(grid, FieldParameters("ayn", velocity_y_m_s=-1, diffusivity_m2_s=0, decay_per_s=0)))):
            self.register_buffer(name, torch.tensor(matrix.toarray(), dtype=torch.float32))
        self.register_buffer("eye", torch.eye(grid.size))
        self.register_buffer("projection", torch.tensor(projection, dtype=torch.float32))
        prior = np.linalg.solve(np.eye(grid.size)-30000**2*self.lap.cpu().numpy(), np.eye(grid.size))
        prior = prior/np.sqrt(np.sum(prior**2)/grid.size)
        self.register_buffer("solar_prior", torch.tensor(prior, dtype=torch.float32))
        self.register_buffer("lower", torch.tril_indices(4, 4))
        default = np.zeros(38, np.float32)
        field = default[:28].reshape(4, 7)
        field[:, 0] = np.arctanh(.05)
        field[:, 1] = np.arctanh(.0125)
        logit = lambda p: np.log(p/(1-p))
        field[:, 2] = logit(2e7/6e8)
        field[:, 3] = logit((.08-.003)/(1.5-.003))
        field[:, 4] = logit((.25-.01)/(2.-.01))
        field[:, 5] = logit((30000.-1000.)/(150000.-1000.))
        if neural:
            layers = [nn.Linear(history*4*grid.size+11, width), nn.SiLU()]
            for _ in range(depth-1):
                layers.extend([nn.Linear(width, width), nn.SiLU()])
            layers.append(nn.Linear(width, 38))
            self.network = nn.Sequential(*layers)
            nn.init.normal_(self.network[-1].weight, std=1e-4)
            with torch.no_grad():
                self.network[-1].bias.copy_(torch.tensor(default))
        else:
            self.constant = nn.Parameter(torch.tensor(default))

    def parameters_at(self, history, calendar):
        output = self.network(torch.cat([history.flatten(1), calendar], dim=1)) if self.neural else self.constant[None].expand(len(history), -1)
        p = output[:, :28].reshape(-1, 4, 7)
        vx, vy = 40000*torch.tanh(p[:, :, 0]), 40000*torch.tanh(p[:, :, 1])
        diffusion = 6e8*torch.sigmoid(p[:, :, 2])
        decay = .003+1.497*torch.sigmoid(p[:, :, 3])
        sigma = .01+1.99*torch.sigmoid(p[:, :, 4])
        length = 1000+149000*torch.sigmoid(p[:, :, 5])
        offset = 1.5*torch.tanh(p[:, :, 6])
        raw = output[:, 28:]
        factor = torch.zeros((len(history), 4, 4), device=output.device, dtype=output.dtype)
        factor[:, self.lower[0], self.lower[1]] = 2*torch.tanh(raw)
        diag = torch.arange(4, device=output.device)
        factor[:, diag, diag] = TF.softplus(raw[:, [0, 2, 5, 9]])+.3
        factor = factor/torch.linalg.vector_norm(factor, dim=-1, keepdim=True)
        return vx, vy, diffusion, decay, sigma, length, offset, factor

    def matrices(self, history, calendar, state=None):
        vx, vy, diffusion, decay, sigma, length, offset, factor = self.parameters_at(history, calendar)
        split = lambda v: ((torch.sqrt(v*v+1.)+v)/2, (torch.sqrt(v*v+1.)-v)/2)
        xp, xn = split(vx); yp, yn = split(vy)
        expand = lambda a: a[:, :, None, None]
        operator = (expand(diffusion)*self.lap+expand(xp)*self.axp+expand(xn)*self.axn
                    +expand(yp)*self.ayp+expand(yn)*self.ayn-expand(decay)*self.eye)
        matrix = self.eye-operator
        smooth = torch.linalg.solve(self.eye-expand(length**2)*self.lap, self.eye.expand(len(history), 4, -1, -1))
        norm = torch.sqrt(torch.sum(smooth**2, dim=(-2, -1))/self.size)
        g = torch.linalg.solve(matrix, smooth)*expand(sigma/norm)
        previous = history[:, -1] if state is None else state
        right = previous+decay[:, :, None]*offset[:, :, None]
        mean = torch.linalg.solve(matrix, right.unsqueeze(-1)).squeeze(-1)
        return mean, g, factor

    def sample_step(self, history, calendar, noise=None, state=None):
        mean, g, factor = self.matrices(history, calendar, state)
        if noise is None:
            noise = torch.randn_like(mean)
        correlated = factor@noise
        return mean+(g@correlated.unsqueeze(-1)).squeeze(-1)

    def negative_log_likelihood(self, history, calendar, target, mask):
        mean, g, factor = self.matrices(history, calendar)
        correlation = factor@factor.transpose(-1, -2)
        covariance = torch.einsum("bfik,bgjk,bfg->bfigj", g, g, correlation).reshape(-1, 4*self.size, 4*self.size)
        valid = mask.reshape(len(history), -1)
        residual = ((target-mean)*mask).reshape(len(history), -1)
        covariance = covariance*valid[:, :, None]*valid[:, None, :]
        # Dummy unit-variance missing channels drop out of logdet and quadratic form.
        diagonal = 1-valid+.025**2*valid
        covariance = covariance+torch.diag_embed(diagonal)
        chol = torch.linalg.cholesky(covariance)
        solved = torch.cholesky_solve(residual[:, :, None], chol).squeeze(-1)
        loss = .5*(torch.sum(residual*solved, dim=1)+2*torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=1))
        return (loss/valid.sum(dim=1).clamp_min(1)).mean()

    def update_history(self, history, state, daylight):
        # Coefficient encoder always uses the same held-cell masking convention.
        observed = state@self.projection.T
        observed = torch.cat([torch.where(daylight[:, None].bool(), observed[:, :1], 0.), observed[:, 1:]], dim=1)
        return torch.cat([history[:, 1:], observed[:, None]], dim=1)

    def rollout(self, history, calendar, daylight, *, members=4):
        batch, steps = calendar.shape[:2]
        history = history.repeat_interleave(members, dim=0)
        state = history[:, -1].clone()
        hidden_solar = (state[:, 0].abs().sum(dim=-1) < 1e-7)
        prior = torch.randn_like(state[:, 0])@self.solar_prior.T
        state = torch.cat([torch.where(hidden_solar[:, None], prior, state[:, 0])[:, None], state[:, 1:]], dim=1)
        paths = []
        for t in range(steps):
            features = calendar[:, t].repeat_interleave(members, dim=0)
            state = self.sample_step(history, features, state=state)
            paths.append(state.reshape(batch, members, 4, self.size))
            history = self.update_history(history, state, daylight[:, t].repeat_interleave(members, dim=0))
        return torch.stack(paths, dim=2)


def rollout_loss(paths, target, mask):
    """Finite-ensemble CRPS and temporal variogram; no point-MSE-only training."""
    members = paths.shape[1]
    error = abs(paths-target[:, None]).mean(dim=1)
    pairs = abs(paths-torch.roll(paths, 1, dims=1)).mean(dim=1)/2
    crps = ((error-pairs)*mask).sum()/mask.sum().clamp_min(1)
    pair_mask = mask[:, 1:]*mask[:, :-1]
    expected = torch.sqrt(abs(paths[:, :, 1:]-paths[:, :, :-1])+1e-6).mean(dim=1)
    truth = torch.sqrt(abs(target[:, 1:]-target[:, :-1])+1e-6)
    variogram = (((expected-truth)**2)*pair_mask).sum()/pair_mask.sum().clamp_min(1)
    return crps+.1*variogram
