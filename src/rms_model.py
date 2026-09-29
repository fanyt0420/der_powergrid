"""QSTS-to-RMS handoff and dynamic DER models on an algebraic network backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from cmath import exp, phase, rect
from dataclasses import dataclass, field
import json
from math import ceil, pi, radians, sin
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.power_flow import DERCommand, NetworkModel, OperatingPoint, PFResult, PFState, PowerFlowSolver, SolverContext


@dataclass(frozen=True)
class DynamicOperatingPoint:
    time_index: int
    bus_voltages: Mapping[str, complex]
    load_pq: Mapping[str, complex]
    der_pq: Mapping[str, DERCommand]
    der_states: Mapping[str, Mapping[str, float]]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RMSState:
    time: float
    bus_voltages: Mapping[str, complex]
    device_states: Mapping[str, Mapping[str, float]]
    pf_state: PFState | None = None


@dataclass(frozen=True)
class RMSResult:
    summary: pd.DataFrame
    bus_voltages: pd.DataFrame
    final_state: RMSState


def _clip_current(p: float, q: float, voltage: float, limit: float) -> tuple[float, float]:
    magnitude = (p * p + q * q) ** 0.5 / max(voltage, 1e-6)
    if magnitude <= limit:
        return p, q
    scale = limit / magnitude
    return p * scale, q * scale


def _bus_voltage(voltages: Mapping[str, complex], bus: str) -> tuple[float, float]:
    """Positive-sequence voltage in p.u. and rad from phase phasors."""
    values = {int(key.rsplit(".", 1)[1]): value for key, value in voltages.items() if key.lower().startswith(f"{bus.lower()}.")}
    if {1, 2, 3}.issubset(values):
        a = exp(2j * pi / 3)
        value = (values[1] + a * values[2] + a * a * values[3]) / 3
    elif values:
        value = next(iter(values.values()))
    else:
        raise ValueError(f"No voltage found for bus {bus!r}.")
    return abs(value), phase(value)


class RMSDevice(ABC):
    """Dynamic DER interface; outputs are solver-neutral total P/Q commands."""

    def __init__(self, device_id: str, bus: str, rating_kw: float, config: Mapping[str, Any], command: DERCommand) -> None:
        self.device_id, self.bus, self.rating_kw, self.config = device_id, bus, rating_kw, config
        self.p_target_kw, self.q_target_kvar = command.p_kw, command.q_kvar
        self.current_limit_pu = float(config["current_limit_pu"])
        if rating_kw <= 0 or self.current_limit_pu <= 0:
            raise ValueError(f"{device_id}: rating_kw and current_limit_pu must be positive.")

    def targets_pu(self) -> tuple[float, float]:
        return self.p_target_kw / self.rating_kw, self.q_target_kvar / self.rating_kw

    def trip(self) -> None:
        self.p_target_kw = self.q_target_kvar = 0.0

    @abstractmethod
    def initialize(self, command: DERCommand, voltage: float, angle: float) -> dict[str, float]: ...

    @abstractmethod
    def derivative(self, state: Mapping[str, float], voltage: float, angle: float) -> dict[str, float]: ...

    @abstractmethod
    def injection(self, state: Mapping[str, float], voltage: float, template: DERCommand) -> DERCommand: ...

    def metrics(self, state: Mapping[str, float], voltage: float, angle: float) -> Mapping[str, float]:
        return {}


class GFLInverter(RMSDevice):
    """SRF-PLL, P/Q outer loops, current inner loop and circular current limit."""

    def __init__(self, *args: Any, nominal_frequency_hz: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.f_nom = nominal_frequency_hz
        for name in ("pll_kp_hz_per_rad", "pll_ki_hz_per_rad_s", "tau_p_control_s", "tau_q_control_s", "tau_current_s"):
            if float(self.config[name]) <= 0:
                raise ValueError(f"{self.device_id}: {name} must be positive.")

    def initialize(self, command: DERCommand, voltage: float, angle: float) -> dict[str, float]:
        p, q = command.p_kw / self.rating_kw, command.q_kvar / self.rating_kw
        return {"p_filter_pu": p, "q_filter_pu": q, "i_d_pu": p / max(voltage, 1e-6), "i_q_pu": q / max(voltage, 1e-6), "pll_angle_rad": angle, "pll_integrator": 0.0}

    def derivative(self, state: Mapping[str, float], voltage: float, angle: float) -> dict[str, float]:
        p_ref, q_ref = self.targets_pu()
        error = sin(angle - float(state["pll_angle_rad"]))
        frequency = float(self.config["pll_kp_hz_per_rad"]) * error + float(self.config["pll_ki_hz_per_rad_s"]) * float(state["pll_integrator"])
        i_d_ref, i_q_ref = _clip_current(float(state["p_filter_pu"]) / max(voltage, 1e-6), float(state["q_filter_pu"]) / max(voltage, 1e-6), voltage, self.current_limit_pu)
        return {"p_filter_pu": (p_ref - float(state["p_filter_pu"])) / float(self.config["tau_p_control_s"]), "q_filter_pu": (q_ref - float(state["q_filter_pu"])) / float(self.config["tau_q_control_s"]), "i_d_pu": (i_d_ref - float(state["i_d_pu"])) / float(self.config["tau_current_s"]), "i_q_pu": (i_q_ref - float(state["i_q_pu"])) / float(self.config["tau_current_s"]), "pll_angle_rad": 2 * pi * frequency, "pll_integrator": error}

    def injection(self, state: Mapping[str, float], voltage: float, template: DERCommand) -> DERCommand:
        p, q = _clip_current(float(state["i_d_pu"]) * voltage, float(state["i_q_pu"]) * voltage, voltage, self.current_limit_pu)
        return template.with_pq(p * self.rating_kw, q * self.rating_kw)

    def metrics(self, state: Mapping[str, float], voltage: float, angle: float) -> Mapping[str, float]:
        error = sin(angle - float(state["pll_angle_rad"]))
        return {"frequency_hz": self.f_nom + float(self.config["pll_kp_hz_per_rad"]) * error + float(self.config["pll_ki_hz_per_rad_s"]) * float(state["pll_integrator"])}


class GFMInverter(RMSDevice):
    """Virtual-inertia grid former with P-f and Q-V droop, voltage loop and limit."""

    def __init__(self, *args: Any, nominal_frequency_hz: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.f_nom = nominal_frequency_hz
        for name in ("inertia_s", "damping_pu_per_hz", "p_droop_pu_per_hz", "q_droop_pu_per_pu", "tau_power_control_s", "tau_power_measure_s", "tau_voltage_control_s"):
            if float(self.config[name]) <= 0:
                raise ValueError(f"{self.device_id}: {name} must be positive.")

    def initialize(self, command: DERCommand, voltage: float, angle: float) -> dict[str, float]:
        p, q = command.p_kw / self.rating_kw, command.q_kvar / self.rating_kw
        return {"p_output_pu": p, "q_output_pu": q, "p_measure_pu": p, "q_measure_pu": q, "frequency_deviation_hz": 0.0, "angle_rad": angle, "voltage_internal_pu": voltage, "voltage_reference_pu": voltage}

    def derivative(self, state: Mapping[str, float], voltage: float, angle: float) -> dict[str, float]:
        p_ref, q_ref = self.targets_pu()
        frequency = float(state["frequency_deviation_hz"])
        p_command = p_ref - float(self.config["p_droop_pu_per_hz"]) * frequency
        q_command = q_ref + float(self.config["q_droop_pu_per_pu"]) * (float(state["voltage_reference_pu"]) - voltage)
        p_command, q_command = _clip_current(p_command, q_command, voltage, self.current_limit_pu)
        voltage_target = float(state["voltage_reference_pu"]) + float(self.config.get("voltage_droop_pu_per_pu", 0.0)) * (q_ref - float(state["q_measure_pu"]))
        return {"p_output_pu": (p_command - float(state["p_output_pu"])) / float(self.config["tau_power_control_s"]), "q_output_pu": (q_command - float(state["q_output_pu"])) / float(self.config["tau_power_control_s"]), "p_measure_pu": (float(state["p_output_pu"]) - float(state["p_measure_pu"])) / float(self.config["tau_power_measure_s"]), "q_measure_pu": (float(state["q_output_pu"]) - float(state["q_measure_pu"])) / float(self.config["tau_power_measure_s"]), "frequency_deviation_hz": (p_ref - float(state["p_measure_pu"]) - float(self.config["damping_pu_per_hz"]) * frequency) / (2 * float(self.config["inertia_s"])), "angle_rad": 2 * pi * frequency, "voltage_internal_pu": (voltage_target - float(state["voltage_internal_pu"])) / float(self.config["tau_voltage_control_s"])}

    def injection(self, state: Mapping[str, float], voltage: float, template: DERCommand) -> DERCommand:
        p, q = _clip_current(float(state["p_output_pu"]), float(state["q_output_pu"]), voltage, self.current_limit_pu)
        return template.with_pq(p * self.rating_kw, q * self.rating_kw)

    def metrics(self, state: Mapping[str, float], voltage: float, angle: float) -> Mapping[str, float]:
        return {"frequency_hz": self.f_nom + float(state["frequency_deviation_hz"]), "voltage_internal_pu": float(state["voltage_internal_pu"])}


class AggregateDER(RMSDevice):
    """Aggregate DER P/Q controls, measurement states, voltage support and limit."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for name in ("tau_p_control_s", "tau_q_control_s", "tau_measure_s"):
            if float(self.config[name]) <= 0:
                raise ValueError(f"{self.device_id}: {name} must be positive.")

    def initialize(self, command: DERCommand, voltage: float, angle: float) -> dict[str, float]:
        p, q = command.p_kw / self.rating_kw, command.q_kvar / self.rating_kw
        return {"p_output_pu": p, "q_output_pu": q, "p_measure_pu": p, "q_measure_pu": q, "voltage_reference_pu": voltage}

    def derivative(self, state: Mapping[str, float], voltage: float, angle: float) -> dict[str, float]:
        p_ref, q_ref = self.targets_pu()
        q_ref += float(self.config.get("voltage_support_pu_per_pu", 0.0)) * (float(state["voltage_reference_pu"]) - voltage)
        p_ref, q_ref = _clip_current(p_ref, q_ref, voltage, self.current_limit_pu)
        return {"p_output_pu": (p_ref - float(state["p_output_pu"])) / float(self.config["tau_p_control_s"]), "q_output_pu": (q_ref - float(state["q_output_pu"])) / float(self.config["tau_q_control_s"]), "p_measure_pu": (float(state["p_output_pu"]) - float(state["p_measure_pu"])) / float(self.config["tau_measure_s"]), "q_measure_pu": (float(state["q_output_pu"]) - float(state["q_measure_pu"])) / float(self.config["tau_measure_s"])}

    def injection(self, state: Mapping[str, float], voltage: float, template: DERCommand) -> DERCommand:
        p, q = _clip_current(float(state["p_output_pu"]), float(state["q_output_pu"]), voltage, self.current_limit_pu)
        return template.with_pq(p * self.rating_kw, q * self.rating_kw)


MODEL_TYPES: Mapping[str, type[RMSDevice]] = {"gfl_inverter": GFLInverter, "gfm_inverter": GFMInverter, "aggregate_der": AggregateDER}


def load_rms_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if "dynamic_devices" not in config or not isinstance(config.get("scenarios"), Mapping):
        raise ValueError("rms_config.json requires dynamic_devices and scenarios.")
    return config


def load_qsts_operating_point(data_dir: str | Path, qsts_output_dir: str | Path, hour: int, network: NetworkModel | None = None) -> DynamicOperatingPoint:
    data_dir, qsts_output_dir = Path(data_dir), Path(qsts_output_dir)
    network = network or NetworkModel.from_json(data_dir / "network.json")
    loads_df, volts_df, system_df = pd.read_csv(data_dir / "load_profiles.csv"), pd.read_csv(qsts_output_dir / "qsts_bus_voltages.csv"), pd.read_csv(qsts_output_dir / "qsts_system.csv")
    rows = system_df[system_df.hour == hour]
    if len(rows) != 1:
        raise ValueError(f"Expected one QSTS result row for hour {hour}, found {len(rows)}.")
    row = rows.iloc[0]
    loads = {str(r.load_name): complex(float(r.p_kw), float(r.q_kvar)) for r in loads_df[loads_df.hour == hour].itertuples(index=False)}
    if set(loads) != network.base_load_names():
        raise ValueError(f"Load data at hour {hour} does not match network.json.")
    voltages = {f"{r.bus}.{int(r.node)}": rect(float(r.v_pu), radians(float(r.angle_deg))) for r in volts_df[volts_df.hour == hour].itertuples(index=False)}
    if not voltages:
        raise ValueError(f"No QSTS bus voltage records at hour {hour}.")
    commands: dict[str, DERCommand] = {}
    slow_states: dict[str, dict[str, float]] = {}
    for device in network.devices:
        if device["kind"] == "load":
            continue
        name, prefix = str(device["id"]), str(device["id"]).lower()
        p_col, q_col = f"{prefix}_p_kw", f"{prefix}_q_kvar"
        if p_col not in row or q_col not in row:
            raise ValueError(f"QSTS output lacks P/Q for {name}.")
        parameters: dict[str, float] = {}
        soc_col = f"{prefix}_soc_pct"
        if soc_col in row and not pd.isna(row[soc_col]):
            parameters["soc_pct"] = float(row[soc_col]); slow_states[name] = {"soc_pct": float(row[soc_col])}
        commands[name] = DERCommand(float(row[p_col]), float(row[q_col]), "qsts_final", parameters)
    return DynamicOperatingPoint(hour, voltages, loads, commands, slow_states, {key: value for key, value in row.to_dict().items() if key != "hour"})


def initialize_rms(operating_point: DynamicOperatingPoint, rms_config: Mapping[str, Any], network: NetworkModel) -> tuple[RMSState, dict[str, RMSDevice]]:
    definitions = {str(d["id"]): d for d in network.devices if d["kind"] != "load"}
    f_nom = float(rms_config.get("nominal_frequency_hz", network.base.get("frequency_hz", 60.0)))
    devices: dict[str, RMSDevice] = {}
    states: dict[str, Mapping[str, float]] = {}
    for config in rms_config["dynamic_devices"]:
        name, model_type = str(config["device_id"]), str(config["model_type"])
        if name in devices or name not in definitions or model_type not in MODEL_TYPES:
            raise ValueError(f"Invalid RMS device configuration: {name!r}, {model_type!r}.")
        definition, command = definitions[name], operating_point.der_pq[name]
        args = (name, str(definition["bus"]), float(definition["p_rated_kw"]), config, command)
        device = MODEL_TYPES[model_type](*args, nominal_frequency_hz=f_nom) if model_type != "aggregate_der" else AggregateDER(*args)
        voltage, angle = _bus_voltage(operating_point.bus_voltages, device.bus)
        devices[name] = device
        states[name] = {**device.initialize(command, voltage, angle), **operating_point.der_states.get(name, {})}
    return RMSState(0.0, dict(operating_point.bus_voltages), states, PFState(operating_point.bus_voltages)), devices


def _events(events: Sequence[Mapping[str, Any]], applied: set[int], time_s: float, loads: dict[str, complex], devices: Mapping[str, RMSDevice]) -> list[str]:
    labels = []
    for index, event in enumerate(events):
        if index in applied or time_s + 1e-12 < float(event["time_s"]):
            continue
        kind, target = str(event["type"]), str(event["device_id"])
        if kind == "load_scale":
            if target not in loads: raise ValueError(f"Unknown base load {target!r}.")
            value = loads[target]; loads[target] = complex(value.real * float(event.get("p_multiplier", 1)), value.imag * float(event.get("q_multiplier", 1)))
        elif kind == "der_trip":
            if target not in devices: raise ValueError(f"DER trip requires a dynamic device: {target!r}.")
            devices[target].trip()
        else: raise ValueError(f"Unsupported RMS event {kind!r}.")
        applied.add(index); labels.append(f"{kind}:{target}")
    return labels


def _rows(time_s: float, result: PFResult, states: Mapping[str, Mapping[str, float]], devices: Mapping[str, RMSDevice], event: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    voltage_rows = [{"time_s": time_s, **record} for record in result.bus_voltage_records]
    summary: dict[str, Any] = {"time_s": time_s, "event": event, "pf_iterations": result.iterations, **result.summary}
    for name, state in states.items():
        voltage, angle = _bus_voltage(result.bus_voltages, devices[name].bus)
        command = devices[name].injection(state, voltage, DERCommand(0, 0, ""))
        prefix = name.lower(); summary[f"{prefix}_p_kw"], summary[f"{prefix}_q_kvar"] = command.p_kw, command.q_kvar
        for key, value in {**state, **devices[name].metrics(state, voltage, angle)}.items():
            if key not in {"p_output_pu", "q_output_pu", "i_d_pu", "i_q_pu"}: summary[f"{prefix}_{key}"] = value
    return voltage_rows, summary


def run_rms(solver: PowerFlowSolver, network: NetworkModel, operating_point: DynamicOperatingPoint, rms_config: Mapping[str, Any], scenario_name: str) -> RMSResult:
    if scenario_name not in rms_config["scenarios"]:
        raise ValueError(f"Unknown RMS scenario {scenario_name!r}.")
    scenario = rms_config["scenarios"][scenario_name]
    dt, end = float(scenario["dt_s"]), float(scenario["t_end_s"])
    if dt <= 0 or end <= 0: raise ValueError("dt_s and t_end_s must be positive.")
    state, devices = initialize_rms(operating_point, rms_config, network)
    loads, commands = dict(operating_point.load_pq), dict(operating_point.der_pq)
    result = solver.solve(OperatingPoint(operating_point.time_index, loads, commands, {"rms_time_s": 0}), state.pf_state, SolverContext())
    state = RMSState(0, result.bus_voltages, state.device_states, result.next_state)
    voltage_rows, summary_rows = [], []
    rows, summary = _rows(0, result, state.device_states, devices, ""); voltage_rows.extend(rows); summary_rows.append(summary)
    applied: set[int] = set()
    for step in range(1, ceil(end / dt) + 1):
        time_s = min(step * dt, end)
        labels = _events(scenario.get("events", []), applied, time_s, loads, devices)
        new_states: dict[str, dict[str, float]] = {}
        for name, device in devices.items():
            voltage, angle = _bus_voltage(state.bus_voltages, device.bus)
            old, derivative = state.device_states[name], device.derivative(state.device_states[name], voltage, angle)
            new_states[name] = dict(old)
            for key, value in derivative.items(): new_states[name][key] = float(old[key]) + (time_s - state.time) * value
            commands[name] = device.injection(new_states[name], voltage, operating_point.der_pq[name])
        result = solver.solve(OperatingPoint(operating_point.time_index, loads, commands, {"rms_time_s": time_s, "scenario": scenario_name}), state.pf_state, SolverContext(time_step_hours=(time_s - state.time) / 3600, previous_result=result))
        state = RMSState(time_s, result.bus_voltages, new_states, result.next_state)
        rows, summary = _rows(time_s, result, new_states, devices, ";".join(labels)); voltage_rows.extend(rows); summary_rows.append(summary)
    return RMSResult(pd.DataFrame(summary_rows), pd.DataFrame(voltage_rows), state)
