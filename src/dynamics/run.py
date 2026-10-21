"""Scenario execution for the launcher and basic mixed validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from src.power_flow import NetworkModel
from src.dynamics.model import KuramotoDAE, validate_dynamic_devices
from src.dynamics.network import PassiveNetwork, PHASE_ANGLE
from src.dynamics.snapshot import capture_snapshot, check_network, save_snapshot
from src.dynamics.solver import NumericalFailure, algebraic_solve, equilibrium, modes, step


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent


def write_json(path, value):
    def numeric(value):
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        raise TypeError(f"Unsupported JSON value: {type(value).__name__}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=numeric), encoding="utf-8")


def validate_config(config, network):
    if config["source_boundary"] not in {"frozen_qsts_terminal", "nominal_ideal"}:
        raise ValueError("Unknown source_boundary")
    for key in ("dt_s", "t_end_s", "tolerance", "min_dt_s", "system_base_va", "switch_impedance_multiplier"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["min_dt_s"] > config["dt_s"]:
        raise ValueError("min_dt_s cannot exceed dt_s")
    iterations = config["max_newton_iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("max_newton_iterations must be a positive integer")
    names = {d["id"]: d for d in network.devices}
    validate_dynamic_devices(config, network)
    for events in config["scenarios"].values():
        for event in events:
            name = event["device_id"]
            if name not in names or not 0 <= event["time_s"] <= config["t_end_s"]:
                raise ValueError(f"Invalid event: {event}")
            if event["type"] not in {"load_scale", "der_trip"}:
                raise ValueError(f"Unknown event type: {event['type']}")
            if event["type"] == "load_scale":
                if names[name]["kind"] != "load" or any(not np.isfinite(event[k]) or event[k] < 0 for k in ("p_multiplier", "q_multiplier")):
                    raise ValueError(f"Invalid load scaling: {event}")
            elif names[name]["kind"] == "load":
                raise ValueError("der_trip cannot target a base load")


def run_case(data_dir, output, config, scenario="flat_run", timestamp=None, snapshot=None):
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    network = NetworkModel.from_json(data_dir / "network.json")
    validate_config(config, network)
    if scenario not in config["scenarios"]:
        raise ValueError(f"Unknown scenario {scenario}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("dae_failure.json", "dae_initialization.json", "dae_modes.json", "network_validation.json"):
        (output / name).unlink(missing_ok=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = "unknown", True
    sources = [*PACKAGE_DIR.glob("*.py"), ROOT / "run_dynamic.py",
               ROOT / "src/power_flow.py", ROOT / "src/profile_store.py",
               ROOT / "src/solvers/fem_elements.py", ROOT / "src/solvers/opendss_solver.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    inputs = [data_dir / "network.json", data_dir / "load_profiles.csv", *sorted(data_dir.rglob("*.dss")), *sorted(data_dir.rglob("*.DSS")), *sorted((data_dir / "der_profiles").glob("*.csv"))]
    manifest = {"success": False, "scenario": scenario, "config": config, "repository_commit": revision, "working_tree_dirty": dirty,
                "source_sha256": hashes, "input_sha256": {str(p.relative_to(data_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "time_unit": "seconds relative to QSTS snapshot", "data_dir": str(data_dir), "source_boundary": config["source_boundary"]}
    import scipy, dss
    manifest["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "dss_python": dss.__version__}
    states, buses, devices, diagnostics = [], [], [], []
    model, x, y = None, None, None
    current_time = 0.0
    stage = "snapshot"
    kwargs = {"scales": {}, "tripped": set()}

    def record(t, x, y, info, side="continuous", dt=0.0):
        f, injection, records, load, static = model.evaluate(x, y, **kwargs)
        v = model.voltage(y)
        kcl = model.kcl(x, y, **kwargs)
        source = np.sum(v[model.passive.fixed] * np.conj(model.passive.current(v)[model.passive.fixed] - injection[model.passive.fixed]))
        generated = sum(complex(r["p_kw"], r["q_kvar"]) * 1000 for r in records if r["device_id"] in {d["device_id"] for d in model.dynamic})
        losses = model.passive.losses_va(v)
        mismatch = source + generated + static - load - losses
        ratio = abs(v) / model.passive.vbase
        violations = model.domain_violations(records)
        diagnostic = {"time_s": t, "event_side": side, "dt_s": dt,
                      "kcl_max_pu": float(max(abs(kcl))), "dynamic_residual_max": info.get("dynamic_residual", 0.0),
                      "newton_iterations": info["iterations"], "power_balance_error_w": float(mismatch.real),
                      "source_p_kw": float(source.real / 1000), "load_including_auxiliary_kw": float(load.real / 1000), "network_loss_kw": float(losses.real / 1000),
                      "voltage_min_pu": float(min(ratio)), "voltage_max_pu": float(max(ratio)),
                      "voltage_violation_count": int(np.sum((ratio < network.constraints.get("voltage_min_pu", .95)) | (ratio > network.constraints.get("voltage_max_pu", 1.05)))),
                      "out_of_model_domain": bool(violations)}
        states.append({"time_s": t, "event_side": side, **dict(zip(model.state_names, map(float, x)))})
        buses.extend({"time_s": t, "event_side": side, "bus": node.rsplit(".", 1)[0], "node": int(node.rsplit(".", 1)[1]), "v_pu": float(abs(voltage) / base), "angle_deg": float(np.angle(voltage) * 180 / np.pi)} for node, voltage, base in zip(model.passive.nodes, v, model.passive.vbase))
        devices.extend({"time_s": t, "event_side": side, **{k: float(v) if isinstance(v, np.floating) else v for k, v in r.items()}} for r in records)
        diagnostics.append(diagnostic)
        if max(abs(kcl)) > config["tolerance"]:
            raise NumericalFailure("kcl", max(abs(kcl)), info["iterations"])
        if abs(mismatch.real) / config["system_base_va"] > config["tolerance"]:
            raise NumericalFailure("power_balance", abs(mismatch.real) / config["system_base_va"], info["iterations"])
        if violations:
            raise NumericalFailure("out_of_model_domain", max(r["current_ratio"] for r in violations), info["iterations"])

    try:
        snapshot = snapshot or capture_snapshot(network, data_dir, timestamp)
        manifest["qsts_timestamp"] = snapshot["timestamp"]
        save_snapshot(snapshot, output)
        stage = "network_validation"
        passive = PassiveNetwork(network, snapshot["nodes"], snapshot["taps"], config)
        model = KuramotoDAE(network, passive, snapshot, config)
        report = check_network(passive, snapshot)
        if not report["passed"]:
            write_json(output / "network_validation.json", report)
            raise NumericalFailure(stage, report["assembly_max_relative_error"], 0)
        stage = "static_regression"
        xg, yg = model.initial_guess()
        ypq, pq_info = algebraic_solve(model, xg, yg, all_pq=True)
        vpq = model.voltage(ypq)
        voltage_error = float(max(abs(vpq - snapshot["voltage"]) / passive.vbase))
        angle_error = float(max(abs(np.angle(vpq / snapshot["voltage"]))) * 180 / np.pi)
        report.update(static_voltage_error_pu=voltage_error, static_angle_error_deg=angle_error,
                      static_passed=voltage_error < 1e-6 and angle_error < 1e-4,
                      auxiliary_consumption_kw={n: float(s.real / 1000) for n, s in snapshot["auxiliary_consumption_va"].items()})
        ideal = np.array([passive.vbase[i] * network.source.get("voltage_pu", 1.0) * np.exp(1j * (np.deg2rad(network.source.get("angle_deg", 0.0)) + PHASE_ANGLE[int(passive.nodes[i].rsplit('.', 1)[1])])) for i in passive.fixed])
        report["source_terminal_vs_ideal_max_pu"] = float(max(abs(snapshot["voltage"][passive.fixed] - ideal) / passive.vbase[passive.fixed]))
        report["source_boundary_explanation"] = f"Freeze actual QSTS source terminal phasors; retains same snapshot and {len(passive.free)} free complex voltages. Nominal ideal source differs by the reported amount."
        write_json(output / "network_validation.json", report)
        if not report["static_passed"]:
            raise NumericalFailure(stage, voltage_error, pq_info["iterations"])
        stage = "initialization"
        x, y, init = equilibrium(model)
        write_json(output / "dae_initialization.json", init)
        manifest["state_count"], manifest["algebraic_count"] = model.nx, model.ny
        manifest["device_bases"] = [{k: d[k] for k in ("device_id", "model", "s_ref_va", "v_nom", "i_ref", "z_ref", "e_star") if k in d} for d in model.dynamic]
        write_json(output / "dae_modes.json", modes(model, x, y))
        events = sorted(config["scenarios"][scenario], key=lambda e: e["time_s"])
        record(0, x, y, init, "pre_event" if events and events[0]["time_s"] == 0 else "continuous")
        if init["status"] != "converged":
            raise NumericalFailure("infeasible_or_out_of_domain", init["residual"], init["iterations"])
        event_index = 0
        stage = "integration"
        while current_time < config["t_end_s"] - 1e-12 or event_index < len(events):
            # Apply events at their exact instant with continuous states and a
            # fresh post-event algebraic solve. Never integrate across a jump.
            if event_index < len(events) and abs(events[event_index]["time_s"] - current_time) < 1e-12:
                while event_index < len(events) and abs(events[event_index]["time_s"] - current_time) < 1e-12:
                    event = events[event_index]
                    if event["type"] == "der_trip":
                        kwargs["tripped"].add(event["device_id"])
                    else:
                        kwargs["scales"][event["device_id"]] = (event["p_multiplier"], event["q_multiplier"])
                    event_index += 1
                y, info = algebraic_solve(model, x, y, **kwargs)
                record(current_time, x, y, info, "post_event")
                continue
            next_event = events[event_index]["time_s"] if event_index < len(events) else config["t_end_s"]
            dt = min(config["dt_s"], config["t_end_s"] - current_time, next_event - current_time)
            while True:
                try:
                    xn, yn, info = step(model, x, y, dt, **kwargs)
                    break
                except NumericalFailure:
                    if dt / 2 < config["min_dt_s"]:
                        raise
                    dt /= 2
            f = model.evaluate(xn, yn, **kwargs)[0]
            info["dynamic_residual"] = float(max(abs(xn - x - dt * f)))
            x, y = xn, yn
            current_time = min(current_time + dt, config["t_end_s"])
            if abs(current_time - next_event) < 1e-12:
                current_time = next_event
            if abs(current_time - config["t_end_s"]) < 1e-12:
                current_time = config["t_end_s"]
            record(current_time, x, y, info, "pre_event" if abs(current_time - next_event) < 1e-12 and event_index < len(events) else "continuous", dt)
        manifest["success"] = True
    except (NumericalFailure, ValueError, np.linalg.LinAlgError) as exc:
        failure = {"failure_stage": getattr(exc, "stage", stage), "failure_time_s": current_time,
                   "message": str(exc), "newton_iterations": getattr(exc, "iterations", None),
                   "jacobian_condition_indicator": getattr(exc, "condition", None), "worst_node_phase": None}
        if model is not None and x is not None and y is not None:
            kcl = model.kcl(x, y, **kwargs)
            worst = int(np.argmax(abs(kcl))) % len(model.passive.free)
            failure["worst_node_phase"] = model.passive.nodes[model.passive.free[worst]]
            failure["worst_node_basis"] = "last recorded state; not a failed Newton trial"
            failure["device_violations"] = model.domain_violations(model.evaluate(x, y, **kwargs)[2])
        write_json(output / "dae_failure.json", failure)
        manifest["failure"] = failure
    finally:
        for name, rows in (("dae_state", states), ("dae_bus_voltage", buses), ("dae_der", devices), ("dae_diagnostics", diagnostics)):
            pd.DataFrame(rows).to_csv(output / (name + ".csv"), index=False)
        manifest["completed_time_s"] = current_time
        write_json(output / "dae_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Three-phase Kuramoto-like DAE")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ieee13_unbalanced_der")
    parser.add_argument("--config", type=Path, help="Defaults to <data-dir>/dynamic_config.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario", default="flat_run", help="Configured scenario name, or all")
    parser.add_argument("--timestamp")
    parser.add_argument("--dt", type=float)
    args = parser.parse_args()
    config = json.loads((args.config or args.data_dir / "dynamic_config.json").read_text(encoding="utf-8"))
    if args.dt is not None:
        config["dt_s"] = args.dt
    scenarios = list(config["scenarios"]) if args.scenario == "all" else [args.scenario]
    snapshot = None
    if len(scenarios) > 1:
        network = NetworkModel.from_json(args.data_dir / "network.json")
        validate_config(config, network)
        snapshot = capture_snapshot(network, args.data_dir, args.timestamp)
    success = True
    for scenario in scenarios:
        output = args.output or ROOT / "output" / args.data_dir.name / "dynamic"
        if args.scenario == "all" or args.output is None:
            output = output / scenario
        result = run_case(args.data_dir, output, config, scenario, args.timestamp, snapshot=snapshot)
        print(json.dumps({"scenario": scenario, "success": result["success"], "completed_time_s": result["completed_time_s"], "output": str(output)}, ensure_ascii=False), flush=True)
        success = success and result["success"]
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
