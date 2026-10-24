"""Regional field interpolation, constrained local SPDEs, and per-site physics.

Local parameters and the example layout are assumptions. The local layer adds
uncertainty, rather than claiming to recover unobserved feeder-scale weather.
All states and physical outputs retain the regional hourly time resolution.
"""
from __future__ import annotations

from dataclasses import asdict
import re

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import expit

from src.regional_spde import FIELDS, SOLAR_CAP, calendar_design
from src.weather_spde import FieldParameters, JointWeatherSPDE, RectGrid

EARTH_M = 6_371_000.


def xy_to_latlon(xy: np.ndarray, origin: dict) -> np.ndarray:
    xy = np.asarray(xy, float)
    return np.column_stack((origin["latitude"]+np.rad2deg(xy[:, 1]/EARTH_M),
                            origin["longitude"]+np.rad2deg(xy[:, 0]/(EARTH_M*np.cos(np.deg2rad(origin["latitude"]))))))


def latlon_to_xy(latlon: np.ndarray, origin: dict) -> np.ndarray:
    a = np.asarray(latlon, float)
    return np.column_stack((np.deg2rad(a[:, 1]-origin["longitude"])*EARTH_M*np.cos(np.deg2rad(origin["latitude"])),
                            np.deg2rad(a[:, 0]-origin["latitude"])*EARTH_M))


def read_layout(config: dict) -> pd.DataFrame:
    sites = pd.DataFrame(config["stations"])
    if len(sites) < 2:
        raise ValueError("At least two sites are required for a spatial comparison.")
    for column in ("site_id", "device_id"):
        if sites[column].duplicated().any() or not sites[column].map(lambda s: bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(s)))).all():
            raise ValueError("Site/device IDs must be unique safe alphanumeric identifiers.")
    if not sites.kind.isin(["pv", "wind"]).all() or not set(sites.kind) == {"pv", "wind"}:
        raise ValueError("Layout must contain PV and wind sites only, at least one of each.")
    if {"latitude", "longitude"} <= set(sites):
        ll = sites[["latitude", "longitude"]].to_numpy(float)
    else:
        ll = xy_to_latlon(sites[["east_m", "north_m"]].to_numpy(float), config["coordinate_origin"])
        sites[["latitude", "longitude"]] = ll
    if not np.isfinite(ll).all() or (abs(ll[:, 0]) >= 85).any() or (abs(ll[:, 1]) > 180).any():
        raise ValueError("Use finite geographic coordinates away from the poles.")
    xy = latlon_to_xy(ll, config["coordinate_origin"])
    sites[["local_x_m", "local_y_m"]] = xy
    if not np.isfinite(sites.p_rated_kw.to_numpy(float)).all() or (sites.p_rated_kw <= 0).any():
        raise ValueError("Rated powers must be finite and positive.")
    for row in sites.to_dict("records"):
        if row["kind"] == "pv":
            if not (0 <= row["surface_tilt"] <= 90 and 0 <= row["surface_azimuth"] < 360):
                raise ValueError("PV tilt must be 0..90 and azimuth 0..<360 degrees.")
        else:
            values = np.array([row[k] for k in ("hub_height_m", "shear_exponent", "cut_in_m_s", "rated_speed_m_s", "cut_out_m_s")])
            if (not np.isfinite(values).all() or values[0] <= 0 or not 0 <= values[1] <= 1
                    or not 0 <= values[2] < values[3] < values[4]):
                raise ValueError("Invalid wind height, shear or ordered cut-in/rated/cut-out speeds.")
    return sites


def interpolate_region(report: dict, latent: np.ndarray, times: pd.DatetimeIndex,
                       sites: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Return raw transformed fields (member,time,field,site), with no new noise."""
    if tuple(report["field_order"]) != FIELDS:
        raise ValueError("Regional field order differs from the site model.")
    grid = RectGrid(np.array(report["grid_x_m"]), np.array(report["grid_y_m"]))
    xy = latlon_to_xy(sites[["latitude", "longitude"]].to_numpy(), report["source"]["coordinate_origin"])
    h = grid.observation_matrix(xy).toarray()
    if latent.ndim != 4 or latent.shape[1:] != (len(times), 4, grid.size) or not np.isfinite(latent).all():
        raise ValueError("Regional latent paths must match time, field and grid dimensions.")
    background = calendar_design(times, xy, pd.Timestamp(report["training"]["start"]))
    coef = np.array(report["background"]["coefficients"])
    mean = np.einsum("tik,fk->tfi", background, coef)
    scale = np.array(report["background"]["residual_scales"])
    raw = np.einsum("stfn,in->stfi", latent, h)*scale[None, None, :, None]+mean
    nearest = np.argmin(np.sum((xy[:, None, :]-grid.xy[None, :, :])**2, axis=2), axis=1)
    return raw, {"regional_xy_m": xy.tolist(), "bilinear_weights": h.tolist(),
                 "nearest_regional_node": nearest.tolist(),
                 "distinct_nearest_nodes": len(np.unique(nearest))}


class LocalResidualSPDE:
    """Hourly local residual field with zero patch-mean at every time step.

    The discrete transition is P M^-1, P=I-11'/N. Exact small-grid stationary
    covariances set the RMS marginal std over the specified sites. Thus length
    sensitivity holds average site variance fixed, including interpolation.
    Regional and local driving processes are independent by construction.
    """

    def __init__(self, sites_xy: np.ndarray, config: dict, noise_length_m: float,
                 dt_seconds: float = 3600., strength: float = 1.):
        width, points = float(config["half_width_m"]), int(config["grid_points"])
        if not np.isfinite(width) or width <= 0 or not 3 <= points <= 16:
            raise ValueError("Use a positive half-width and 3..16 local grid points per axis.")
        if not np.isfinite(strength) or strength < 0 or not np.isfinite(noise_length_m) or noise_length_m <= 0:
            raise ValueError("Use positive finite noise length and nonnegative strength.")
        self.grid = RectGrid(np.linspace(-width, width, points), np.linspace(-width, width, points))
        self.h = self.grid.observation_matrix(np.asarray(sites_xy, float)).toarray()
        n = self.grid.size
        projection = np.eye(n)-np.ones((n, n))/n
        fields, marginals, initial_factors, contractions = [], [], [], []
        if tuple(field["name"] for field in config["fields"]) != FIELDS:
            raise ValueError("Local fields must follow regional field order.")
        for field in config["fields"]:
            target = float(field["station_rms_std"])*strength
            if not np.isfinite(target) or target < 0:
                raise ValueError("Local marginal standard deviations must be nonnegative and finite.")
            args = {k: v for k, v in field.items() if k != "station_rms_std"}
            unit = FieldParameters(**args, noise_std_per_sqrt_s=1., noise_length_m=noise_length_m)
            one = JointWeatherSPDE(self.grid, [unit], dt_seconds)
            f, g, _ = one.dense_single_field()
            f, g = projection@f, projection@g
            covariance = solve_discrete_lyapunov(f, g@g.T)
            covariance = (covariance+covariance.T)/2
            site_cov = self.h@covariance@self.h.T
            rms = float(np.sqrt(np.trace(site_cov)/len(self.h)))
            if not np.isfinite(rms) or rms <= 0:
                raise ValueError("Local model cannot support nonconstant spatial variability.")
            amplitude = target/rms
            fields.append(FieldParameters(**args, noise_std_per_sqrt_s=amplitude, noise_length_m=noise_length_m))
            marginals.append(site_cov*amplitude**2)
            values, vectors = np.linalg.eigh(covariance*amplitude**2)
            if values.min() < -1e-7:
                raise ValueError("Local stationary covariance is not positive semidefinite.")
            initial_factors.append(vectors*np.sqrt(np.maximum(values, 0)))
            contractions.append(float(np.linalg.norm(f, 2)))
        self.model = JointWeatherSPDE(self.grid, fields, dt_seconds, np.array(config["driving_correlation"], float))
        self.site_covariances = np.array(marginals)
        self._initial_factors = initial_factors
        if max(contractions) >= 1:
            raise ValueError("Local projected dynamics must contract in the Euclidean norm.")
        # Independent fields have exact marginal stationary initialization.
        # If a user supplies correlated local drivers, burn in from zero until
        # the missing stationary covariance contracts by <=1e-12 in norm.
        self.burnin_steps = (0 if np.allclose(self.model.driving_correlation, np.eye(4)) else
                            int(np.ceil(np.log(1e-6)/np.log(max(contractions)))))
        self.fields = fields

    def sample(self, members: int, steps: int, seed: int) -> tuple[np.ndarray, dict]:
        if members < 2 or steps < 1:
            raise ValueError("Need >=2 members and positive steps.")
        rng = np.random.default_rng(seed)
        state = np.zeros((members, 4, self.grid.size))
        if not self.burnin_steps:
            for f, factor in enumerate(self._initial_factors):
                state[:, f] = rng.standard_normal((members, self.grid.size))@factor.T
        else:
            for _ in range(self.burnin_steps):
                state = self.model.step(state, rng)
                state -= state.mean(axis=-1, keepdims=True)
        paths, mean_error = [], 0.
        for _ in range(steps):
            state = self.model.step(state, rng)
            state -= state.mean(axis=-1, keepdims=True)
            mean_error = max(mean_error, float(abs(state.mean(axis=-1)).max()))
            paths.append(np.einsum("in,sfn->sfi", self.h, state))
        return np.stack(paths, axis=1), {"max_patch_mean_abs": mean_error,
                                        "burnin_steps": self.burnin_steps,
                                        "parameters": [asdict(p) for p in self.fields],
                                        "station_rms_std": np.sqrt(np.trace(self.site_covariances, axis1=1, axis2=2)/len(self.h)).tolist(),
                                        "site_covariances": self.site_covariances.tolist()}


def weather_from_raw(raw: np.ndarray, times: pd.DatetimeIndex, sites: pd.DataFrame,
                     pv_shear: float = .14) -> dict[str, np.ndarray]:
    """raw=(member,time,field,site); radiation remains representative hourly mean."""
    if raw.ndim != 4 or raw.shape[1:] != (len(times), 4, len(sites)) or not np.isfinite(raw).all():
        raise ValueError("Raw fields must match finite member/time/field/site dimensions.")
    if times.tz is None:
        raise ValueError("Use timezone-aware UTC midpoint timestamps.")
    clear = np.zeros((len(times), len(sites)))
    for i, site in enumerate(sites.itertuples()):
        loc = Location(site.latitude, site.longitude, tz="UTC", altitude=100.)
        for offset, weight in zip((-30, -15, 0, 15, 30), (1, 4, 2, 4, 1), strict=True):
            clear[:, i] += weight*loc.get_clearsky(times+np.timedelta64(offset, "m"))["ghi"].to_numpy()/12
    u, v = raw[:, :, 1], raw[:, :, 2]
    speed = np.hypot(u, v)
    return {"ghi": SOLAR_CAP*expit(raw[:, :, 0])*clear, "clear_sky_ghi": clear,
            "clear_sky_index": SOLAR_CAP*expit(raw[:, :, 0]), "temp_air": raw[:, :, 3],
            "wind_u_100m": u, "wind_v_100m": v, "wind_speed_100m": speed,
            "wind_speed": speed*(.1**pv_shear)}


def wind_capacity_factor(speed100: np.ndarray, site: dict) -> np.ndarray:
    speed = np.asarray(speed100, float)*(site["hub_height_m"]/100)**site["shear_exponent"]
    cut_in, rated, cut_out = (site[k] for k in ("cut_in_m_s", "rated_speed_m_s", "cut_out_m_s"))
    ramp = (speed**3-cut_in**3)/(rated**3-cut_in**3)
    return np.where((speed >= cut_in) & (speed < cut_out), np.clip(ramp, 0, 1), 0.)


def site_power(weather: dict, times: pd.DatetimeIndex, sites: pd.DataFrame, config: dict) -> tuple[np.ndarray, dict]:
    """Vectorize ensemble members within each site's PVWatts/ERBS evaluation."""
    members, hours, count = weather["ghi"].shape
    repeated_times = pd.DatetimeIndex(np.tile(times.asi8, members), tz="UTC")
    power, decomposed = np.zeros((members, hours, count)), {}
    for i, site in enumerate(sites.to_dict("records")):
        if site["kind"] == "wind":
            power[:, :, i] = site["p_rated_kw"]*wind_capacity_factor(weather["wind_speed_100m"][:, :, i], site)
            continue
        loc = Location(site["latitude"], site["longitude"], tz="UTC", altitude=100.)
        zenith = loc.get_solarposition(repeated_times)["zenith"].to_numpy()
        radiation = pvlib.irradiance.erbs(weather["ghi"][:, :, i].reshape(-1), zenith, repeated_times)
        frame = pd.DataFrame({"ghi": weather["ghi"][:, :, i].reshape(-1),
                              "dni": radiation.dni.to_numpy(), "dhi": radiation.dhi.to_numpy(),
                              "temp_air": weather["temp_air"][:, :, i].reshape(-1),
                              "wind_speed": weather["wind_speed"][:, :, i].reshape(-1)}, index=repeated_times)
        model = PVSystem(surface_tilt=site["surface_tilt"], surface_azimuth=site["surface_azimuth"],
                         module_parameters={"pdc0": 1., "gamma_pdc": config["pv_gamma_pdc"]},
                         inverter_parameters={"pdc0": 1/(config["pv_dc_ac_ratio"]*config["pv_inverter_efficiency"]),
                                              "eta_inv_nom": config["pv_inverter_efficiency"]},
                         temperature_model_parameters=pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][config["pv_temperature_model"]])
        chain = ModelChain.with_pvwatts(model, loc, aoi_model="physical", spectral_model="no_loss")
        chain.run_model(frame)
        power[:, :, i] = site["p_rated_kw"]*np.maximum(chain.results.ac.to_numpy().reshape(members, hours), 0.)
        decomposed[site["device_id"]] = {"dni": frame.dni.to_numpy().reshape(members, hours),
                                          "dhi": frame.dhi.to_numpy().reshape(members, hours)}
    if not np.isfinite(power).all() or (power < 0).any():
        raise ValueError("Invalid site power conversion.")
    return power, decomposed


def pooled_conditional_correlation(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Member,time,site; remove each hour's ensemble mean before pooling."""
    if mask is not None:
        values = values[:, mask]
    centered = values-values.mean(axis=0, keepdims=True)
    flat = centered.reshape(-1, centered.shape[-1])
    if flat.shape[1] == 1:
        return np.ones((1, 1))
    return np.corrcoef(flat.T)


def aggregation_metrics(power: np.ndarray, rating: np.ndarray) -> dict:
    """Conditional variance ratio vs fully correlated outputs with same marginals."""
    fleet = power.sum(axis=-1)
    variance = np.var(fleet, axis=0, ddof=1)
    reference = np.sum(np.std(power, axis=0, ddof=1), axis=-1)**2
    good = reference > 1e-10
    total = float(np.sum(rating))
    residual = fleet-fleet.mean(axis=0, keepdims=True)
    return {"installed_kw": total, "conditional_variance_ratio_to_perfect_correlation": float(variance[good].sum()/reference[good].sum()),
            "conditional_fleet_std_kw_rms": float(np.sqrt(variance[good].mean())),
            "p95_abs_hourly_ramp_pu": float(np.quantile(abs(np.diff(fleet, axis=1))/total, .95)),
            "p95_abs_centered_hourly_ramp_pu": float(np.quantile(abs(np.diff(residual, axis=1))/total, .95)),
            "positive_variance_hours": int(good.sum())}
