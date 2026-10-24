"""Assumed feeder-site layer over the frozen, fitted regional hourly ensemble.

Example: python data/generate_data/station_spde.py
Outputs a sensitivity study and two isolated QSTS cases with identical devices.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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

import ieee13_era5 as base
from src.spde_case import export_case
from src.site_weather import (LocalResidualSPDE, read_layout, interpolate_region,
                               weather_from_raw, site_power, pooled_conditional_correlation,
                               aggregation_metrics)
from src.weather_spde import implementation_fingerprint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(raw, weather, power, sites):
    xy = sites[["local_x_m", "local_y_m"]].to_numpy()
    distance = np.linalg.norm(xy[:, None]-xy[None, :], axis=-1)
    ii, jj = np.triu_indices(len(sites), 1)
    order = np.argsort(distance[ii, jj])
    pairs = {"nearest": (ii[order[0]], jj[order[0]]), "farthest": (ii[order[-1]], jj[order[-1]])}
    solar = pooled_conditional_correlation(raw[:, :, 0], (weather["clear_sky_ghi"] >= 50).all(axis=1))
    result = {"solar_latent_conditional_correlation": solar.tolist(), "pairs": {}, "groups": {}}
    for name, (i, j) in pairs.items():
        result["pairs"][name] = {"sites": [sites.site_id.iloc[i], sites.site_id.iloc[j]],
                                  "distance_m": float(distance[i, j]), "solar_latent_correlation": float(solar[i, j])}
    for kind in ("pv", "wind"):
        select = np.flatnonzero(sites.kind == kind)
        mask = (weather["clear_sky_ghi"][:, select] >= 50).all(axis=1) if kind == "pv" else None
        matrix = pooled_conditional_correlation(power[:, :, select], mask)
        pair_values = matrix[np.triu_indices(len(select), 1)]
        result["groups"][kind] = {**aggregation_metrics(power[:, :, select], sites.p_rated_kw.to_numpy()[select]),
                                   "power_conditional_correlation": matrix.tolist(),
                                   "pair_correlation_min": float(pair_values.min()) if len(pair_values) else None,
                                   "pair_correlation_max": float(pair_values.max()) if len(pair_values) else None}
    return result


def export_sites(case, times, sites, power, weather, decomposition, member, metadata):
    templates = {d["id"]: d for d in base._ders()}
    devices, frames, prescribed = [], {}, {}
    starts = times-np.timedelta64(30, "m")
    for i, row in enumerate(sites.to_dict("records")):
        if row["template_id"] not in templates or templates[row["template_id"]]["kind"] != row["kind"]:
            raise ValueError("Choose a compatible existing device template for each site's electrical connection.")
        device = dict(templates[row["template_id"]])
        device.update(id=row["device_id"], p_rated_kw=row["p_rated_kw"],
                      s_rated_kva=base.INVERTER_S_RATIO*row["p_rated_kw"],
                      latitude=row["latitude"], longitude=row["longitude"], weather_site_id=row["site_id"],
                      profile_file=f"der_profiles/{row['device_id'].lower()}.csv")
        for key in ("bus", "surface_tilt", "surface_azimuth", "hub_height_m", "shear_exponent",
                    "cut_in_m_s", "rated_speed_m_s", "cut_out_m_s"):
            if key in row and pd.notna(row[key]):
                device[key] = row[key]
        devices.append(device)
        frame = pd.DataFrame({key: weather[key][member, :, i] for key in
                              ("ghi", "temp_air", "wind_speed", "wind_speed_100m", "wind_u_100m", "wind_v_100m")}, index=starts)
        radiation = decomposition.get(row["device_id"])
        frame["dni"] = radiation["dni"][member] if radiation else 0.
        frame["dhi"] = radiation["dhi"][member] if radiation else frame.ghi
        frames[row["device_id"]], prescribed[row["device_id"]] = frame, power[member, :, i]
    export_case(case, starts, frames, sites, devices, metadata, active_power_kw=prescribed)
    pd.concat([frame.assign(device_id=name).rename_axis("timestamp_utc").reset_index()
               for name, frame in frames.items()], ignore_index=True).to_csv(case/"weather_by_site.csv", index=False)


def run(args):
    report_path = args.region/"model.json"
    ensemble_path = args.region/"weather_ensemble_24h.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    local = json.loads(args.local.read_text(encoding="utf-8"))
    if not local.get("parameters_are_assumed"):
        raise ValueError("This pilot only supports explicitly assumed local parameters; calibration needs a separate protocol.")
    sites = read_layout(layout)
    with np.load(ensemble_path) as source:
        latent, times = source["latent"].copy(), pd.to_datetime(source["timestamps"], utc=True)
    if not 0 <= args.member < len(latent):
        raise ValueError("Selected member is outside the upstream ensemble.")
    raw_region, interpolation = interpolate_region(report, latent, times, sites)
    sites["nearest_regional_node"] = interpolation["nearest_regional_node"]
    lengths = [float(value) for value in local["noise_lengths_m"]]
    default = float(local["default_length_m"])
    if len(set(lengths)) != len(lengths) or default not in lengths or any(value <= 0 for value in lengths):
        raise ValueError("Use distinct positive lengths including the default length.")
    strengths = [float(s) for s in local.get("strength_multipliers", [1.])]
    if 1. not in strengths or len(set(strengths)) != len(strengths) or not all(np.isfinite(s) and s > 0 for s in strengths):
        raise ValueError("Use distinct positive finite strength multipliers including 1.")
    selected_name = f"local_{default:g}m"
    case_names = ("regional_only", selected_name)
    cases = {name: ROOT/"data"/f"{args.name_prefix}_{name}" for name in case_names}
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty --output directory for a new experiment.")
    for case in cases.values():
        if case.exists() and any(case.iterdir()):
            raise FileExistsError(f"Case already populated: {case}. Choose a new --name-prefix.")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"status": "station local-SPDE mechanism/sensitivity study; local parameters are assumed, not station-data calibrated",
              "layout": layout, "local_configuration": local, "seed": args.seed,
              "upstream_model_sha256": digest(report_path), "upstream_ensemble_sha256": digest(ensemble_path),
              "layout_sha256": digest(args.layout), "local_configuration_sha256": digest(args.local),
              "regional_interpolation": interpolation, "time_start_midpoint_utc": str(times[0]),
              "time_end_midpoint_utc": str(times[-1]), "members": len(latent), "hours": len(times),
              "scores_are_observation_validation": False, "models": {},
              "clear_sky_altitude_assumption_m": 100.,
              "interpretation": ["Regional ensemble is frozen and identical across all comparisons",
                                 "Local field is independent of regional uncertainty, adding uncalibrated extra variance",
                                 "Patch-mean is constrained in latent units; nonlinear inverse maps need not preserve mean GHI or power",
                                 "Noise length is a smoother parameter, not an empirically calibrated correlation length",
                                 "Local wind curve, heights, PV geometries and all coordinates are assumed",
                                 "No new station-specific mean climate or terrain effects have been fitted",
                                 "All results remain hourly; no minute-scale weather has been inferred"]}
    implementation = implementation_fingerprint()
    for name in ("src/site_weather.py", "data/generate_data/station_spde.py"):
        implementation["source_sha256"][name] = digest(ROOT/name)
    result["implementation"] = implementation
    jobs = [(None, 0.), *[(length, 1.) for length in lengths], *[(default, s) for s in strengths if s != 1.]]
    cached_local = {}
    for length, strength in jobs:
        name = "regional_only" if length is None else f"local_{length:g}m"
        if length is not None and strength != 1.:
            name += f"_strength_{strength:g}"
        print(f"Simulating {name}: {len(latent)} members, {len(sites)} sites, {len(times)} hourly steps", flush=True)
        perturbation, diagnostics = np.zeros_like(raw_region), None
        if length is not None:
            if length not in cached_local:
                model = LocalResidualSPDE(sites[["local_x_m", "local_y_m"]].to_numpy(), local, length)
                cached_local[length] = model.sample(len(latent), len(times), args.seed)
            base_paths, base_diagnostics = cached_local[length]
            perturbation, diagnostics = base_paths*strength, deepcopy(base_diagnostics)
            diagnostics["strength_multiplier"] = strength
            diagnostics["max_patch_mean_abs"] *= strength
            diagnostics["station_rms_std"] = (np.array(diagnostics["station_rms_std"])*strength).tolist()
            diagnostics["site_covariances"] = (np.array(diagnostics["site_covariances"])*strength**2).tolist()
            for field in diagnostics["parameters"]:
                field["noise_std_per_sqrt_s"] *= strength
        raw = raw_region+perturbation
        weather = weather_from_raw(raw, times, sites, local["pv_wind_10m_shear_exponent"])
        power, radiation = site_power(weather, times, sites, local)
        metrics = summary(raw, weather, power, sites)
        metrics["local_model"] = diagnostics
        # Isolate equipment differences using exactly the same per-site weather.
        uniform = sites.copy()
        uniform.loc[uniform.kind == "pv", ["surface_tilt", "surface_azimuth"]] = [30., 180.]
        uniform.loc[uniform.kind == "wind", ["hub_height_m", "shear_exponent", "cut_in_m_s", "rated_speed_m_s", "cut_out_m_s"]] = [100., .14, 3., 12., 25.]
        power_uniform, _ = site_power(weather, times, uniform, local)
        metrics["equipment_ablation_same_weather"] = {}
        for kind in ("pv", "wind"):
            chosen = np.flatnonzero(sites.kind == kind)
            ratings = sites.p_rated_kw.to_numpy()[chosen]
            mask = (weather["clear_sky_ghi"][:, chosen] >= 50).all(axis=1) if kind == "pv" else np.ones(len(times), bool)
            difference = abs(power[:, mask][:, :, chosen]-power_uniform[:, mask][:, :, chosen])/ratings
            metrics["equipment_ablation_same_weather"][kind] = {
                "mean_abs_change_in_capacity_factor": float(difference.mean()),
                "uniform_device_aggregation": aggregation_metrics(power_uniform[:, :, chosen], ratings)}
        result["models"][name] = metrics
        np.savez_compressed(args.output/f"{name}.npz", raw=raw, local_residual=perturbation, power_kw=power,
                            timestamps=times.asi8, site_ids=sites.site_id.to_numpy(str),
                            **{k: weather[k] for k in ("ghi", "temp_air", "wind_speed_100m")})
        if name in cases:
            metadata = {"status": result["status"], "coordinate_origin": layout["coordinate_origin"],
                        "coordinates_are_assumed": layout["coordinates_are_assumed"],
                        "equipment_parameters_are_assumed": layout["equipment_parameters_are_assumed"],
                        "local_parameters_are_assumed": True, "member": args.member, "comparison": name,
                        "weather_geography_source": "configured layout; regional bilinear background plus optional assumed local SPDE; IEEE13 bus mapping is illustrative",
                        "upstream_model_sha256": result["upstream_model_sha256"],
                        "upstream_ensemble_sha256": result["upstream_ensemble_sha256"],
                        "pv_total_kw": float(sites.loc[sites.kind == "pv", "p_rated_kw"].sum()),
                        "wind_total_kw": float(sites.loc[sites.kind == "wind", "p_rated_kw"].sum()),
                        "site_configuration": layout, "local_configuration": local,
                        "power_time_convention": "interval start UTC; representative midpoint physics",
                        "loads": "same synthetic reference shapes in both cases", "implementation": implementation}
            export_sites(cases[name], times, sites, power, weather, radiation, args.member, metadata)
        print(f"  PV variance ratio={metrics['groups']['pv']['conditional_variance_ratio_to_perfect_correlation']:.4f}; "
              f"Wind={metrics['groups']['wind']['conditional_variance_ratio_to_perfect_correlation']:.4f}", flush=True)
    result["exported_cases"] = {name: str(case.resolve()) for name, case in cases.items()}
    sites.to_csv(args.output/"sites.csv", index=False)
    (args.output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plot_result(result, sites, args.output)
    print(f"Station report: {args.output/'summary.json'}", flush=True)


def plot_result(result, sites, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for kind, marker in (("pv", "o"), ("wind", "^")):
        selected = sites[sites.kind == kind]
        axes[0, 0].scatter(selected.local_x_m/1000, selected.local_y_m/1000, marker=marker, label=kind.upper())
        for row in selected.itertuples():
            axes[0, 0].annotate(row.site_id.replace("SITE_", ""), (row.local_x_m/1000, row.local_y_m/1000), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0, 0].set(xlabel="East (km)", ylabel="North (km)",
                   title=f"Configured layout, {result['regional_interpolation']['distinct_nearest_nodes']} nearest ERA5 node(s)")
    axes[0, 0].legend()
    axes[0, 0].margins(x=.15, y=.15)
    xy = sites[["local_x_m", "local_y_m"]].to_numpy()
    distance = np.linalg.norm(xy[:, None]-xy[None], axis=-1)
    i, j = np.triu_indices(len(sites), 1)
    for name, model in result["models"].items():
        local = model["local_model"]
        if local is None or local["strength_multiplier"] != 1.:
            continue
        cov = np.array(local["site_covariances"])[0]
        correlation = cov/np.sqrt(np.diag(cov)[:, None]*np.diag(cov)[None, :])
        axes[0, 1].scatter(distance[i, j]/1000, correlation[i, j], s=15, alpha=.75, label=name)
    axes[0, 1].set(xlabel="Pair distance (km)", ylabel="Local solar latent correlation", title="Theoretical local covariance, fixed mean site variance")
    axes[0, 1].legend(fontsize=8)
    names = list(result["models"])
    for k, kind in enumerate(("pv", "wind")):
        ratios = [result["models"][n]["groups"][kind]["conditional_variance_ratio_to_perfect_correlation"] for n in names]
        axes[1, 0].bar(np.arange(len(names))+(k-.5)*.35, ratios, width=.35, label=kind.upper())
    labels = ["Regional only" if name == "regional_only" else
              name.replace("local_", "Local ").replace("2000m_strength_", "2 km x").replace("1000m", "1 km").replace("2000m", "2 km").replace("4000m", "4 km")
              for name in names]
    axes[1, 0].set_xticks(np.arange(len(names)), labels, rotation=25)
    axes[1, 0].set(ylabel="Variance / perfectly correlated reference", title="Same capacities and devices in every weather case")
    axes[1, 0].legend()
    chosen = f"local_{result['local_configuration']['default_length_m']:g}m"
    sample = np.load(output/f"{chosen}.npz")
    times = pd.to_datetime(sample["timestamps"], utc=True)
    pv = np.flatnonzero(sites.kind == "pv")
    for column in pv:
        axes[1, 1].plot(times, sample["power_kw"][0, :, column]/sites.p_rated_kw.iloc[column], lw=1, label=sites.site_id.iloc[column].replace("SITE_", ""))
    axes[1, 1].set(ylabel="PV power / DC rating", xlabel="UTC midpoint", title="One member: site weather + different equipment")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].legend(fontsize=7, ncol=2)
    fig.suptitle("Assumed station-layer sensitivity over frozen regional ERA5 ensemble; hourly")
    fig.tight_layout()
    fig.savefig(output/"station_sensitivity.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=Path, default=ROOT/"data/processed/spde/regional_driver")
    parser.add_argument("--layout", type=Path, default=ROOT/"data/config/spde_station_example.json")
    parser.add_argument("--local", type=Path, default=ROOT/"data/config/spde_local_example.json")
    parser.add_argument("--output", type=Path, default=ROOT/"output/spde/stations_v2")
    parser.add_argument("--name-prefix", default="ieee13_spde_sites_v2")
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if Path(args.name_prefix).name != args.name_prefix or not args.name_prefix.startswith("ieee13_spde_"):
        parser.error("Use a simple --name-prefix beginning with ieee13_spde_.")
    run(args)


if __name__ == "__main__":
    main()
