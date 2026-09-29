"""QSTS-to-RMS handoff and a minimal solver-neutral RMS simulation framework.

The first RMS model deliberately keeps the network quasi-steady algebraic:
each time step integrates device states and calls a :class:`PowerFlowSolver` to
enforce the network equations.  It is an architectural stepping stone for a
future DAE solver, not a replacement for an electromagnetic-transient model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.power_flow import DERCommand, NetworkModel, OperatingPoint, PFResult, PFState, PowerFlowSolver, SolverContext


@dataclass(frozen=True)
class DynamicOperatingPoint:
    """Complete solved QSTS operating point used to initialize RMS."""

    time_index: int
    bus_voltages: Mapping[str, complex]
    load_pq: Mapping[str, complex]
    der_pq: Mapping[str, DERCommand]
    der_states: Mapping[str, Mapping[str, float]]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RMSState:
    """RMS DAE state at one time instant: dynamic device state ``x`` and bus ``z``."""

    time: float
    bus_voltages: Mapping[str, complex]
    device_states: Mapping[str, Mapping[str, float]]
    pf_state: PFState | None = None


@dataclass(frozen=True)
class RMSResult:
    """In-memory RMS output. Data frames are written by ``run_rms.py`` only."""

    summary: pd.DataFrame
    bus_voltages: pd.DataFrame
    final_state: RMSState


@dataclass
class RMSDevice:
    """Minimal first-order P/Q tracking dynamic model for one existing DER.

    ``p_kw`` and ``q_kvar`` are the dynamic states.  Their targets come from the
    selected QSTS operating point and may be altered by a disturbance.
    """

    device_id: str
    model_type: str
    tau_p_s: float
    tau_q_s: float
    p_target_kw: float
    q_target_kvar: float

    def initialize(self, command: DERCommand) -> dict[str, float]:
        return {"p_kw": command.p_kw, "q_kvar": command.q_kvar}

    def derivative(self, state: Mapping[str, float]) -> dict[str, float]:
        if self.model_type != "first_order_pq":
            raise ValueError(f"Unsupported RMS model type {self.model_type!r} for {self.device_id}.")
        return {
            "p_kw": (self.p_target_kw - float(state["p_kw"])) / self.tau_p_s,
            "q_kvar": (self.q_target_kvar - float(state["q_kvar"])) / self.tau_q_s,
        }

    def injection(self, state: Mapping[str, float], template: DERCommand) -> DERCommand:
        return template.with_pq(float(state["p_kw"]), float(state["q_kvar"]))

    def trip(self) -> None:
        self.p_target_kw = 0.0
        self.q_target_kvar = 0.0


def _require_unique_row(frame: pd.DataFrame, column: str, value: int) -> pd.Series:
    rows = frame[frame[column] == value]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {column}={value} row, found {len(rows)}.")
    return rows.iloc[0]


def load_rms_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate RMS-only model parameters and events."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if "scenarios" not in config or not isinstance(config["scenarios"], Mapping):
        raise ValueError("rms_config.json must define a 'scenarios' object.")
    if "dynamic_devices" not in config:
        raise ValueError("rms_config.json must define 'dynamic_devices'.")
    return config


def load_qsts_operating_point(
    data_dir: str | Path,
    qsts_output_dir: str | Path,
    hour: int,
    network: NetworkModel | None = None,
) -> DynamicOperatingPoint:
    """Reconstruct the solved operating point at one QSTS hour from saved output."""
    data_dir, qsts_output_dir = Path(data_dir), Path(qsts_output_dir)
    network = network or NetworkModel.from_json(data_dir / "network.json")
    load_profile = pd.read_csv(data_dir / "load_profiles.csv")
    voltage_data = pd.read_csv(qsts_output_dir / "qsts_bus_voltages.csv")
    system_data = pd.read_csv(qsts_output_dir / "qsts_system.csv")
    system_row = _require_unique_row(system_data, "hour", hour)
    hour_loads = load_profile[load_profile["hour"] == hour]
    expected_loads = network.base_load_names()
    loads = {str(row.load_name): complex(float(row.p_kw), float(row.q_kvar)) for row in hour_loads.itertuples(index=False)}
    if set(loads) != expected_loads:
        raise ValueError(f"QSTS load profile at hour {hour} does not match network loads.")

    voltages: dict[str, complex] = {}
    for row in voltage_data[voltage_data["hour"] == hour].itertuples(index=False):
        if pd.isna(row.v_pu) or pd.isna(row.angle_deg):
            raise ValueError(f"Invalid saved voltage at hour {hour}, bus={row.bus}, node={row.node}.")
        from cmath import rect
        from math import radians
        voltages[f"{row.bus}.{int(row.node)}"] = rect(float(row.v_pu), radians(float(row.angle_deg)))
    if not voltages:
        raise ValueError(f"No saved bus voltages for QSTS hour {hour}.")

    commands: dict[str, DERCommand] = {}
    states: dict[str, dict[str, float]] = {}
    for device in network.devices:
        if device["kind"] == "load":
            continue
        device_id = str(device["id"])
        prefix = device_id.lower()
        p_column, q_column = f"{prefix}_p_kw", f"{prefix}_q_kvar"
        if p_column not in system_row or q_column not in system_row:
            raise ValueError(f"QSTS system output lacks final P/Q columns for {device_id}.")
        parameters: dict[str, float] = {}
        soc_column = f"{prefix}_soc_pct"
        if soc_column in system_row and not pd.isna(system_row[soc_column]):
            parameters["soc_pct"] = float(system_row[soc_column])
            states[device_id] = {"soc_pct": float(system_row[soc_column])}
        commands[device_id] = DERCommand(float(system_row[p_column]), float(system_row[q_column]), "qsts_final", parameters)

    metadata = {key: value for key, value in system_row.to_dict().items() if key not in {"hour"}}
    return DynamicOperatingPoint(hour, voltages, loads, commands, states, metadata)


def initialize_rms(
    operating_point: DynamicOperatingPoint,
    rms_config: Mapping[str, Any],
) -> tuple[RMSState, dict[str, RMSDevice]]:
    """Set x(0), z(0) directly from a solved QSTS operating point."""
    devices: dict[str, RMSDevice] = {}
    states: dict[str, Mapping[str, float]] = {}
    for raw in rms_config["dynamic_devices"]:
        device_id = str(raw["device_id"])
        if device_id not in operating_point.der_pq:
            raise ValueError(f"RMS device {device_id!r} is not a DER in the QSTS operating point.")
        tau_p, tau_q = float(raw["tau_p_s"]), float(raw["tau_q_s"])
        if tau_p <= 0 or tau_q <= 0:
            raise ValueError(f"RMS time constants for {device_id} must be positive.")
        command = operating_point.der_pq[device_id]
        device = RMSDevice(device_id, str(raw["model_type"]), tau_p, tau_q, command.p_kw, command.q_kvar)
        devices[device_id] = device
        states[device_id] = device.initialize(command)
    return RMSState(0.0, dict(operating_point.bus_voltages), states, PFState(operating_point.bus_voltages)), devices


def _apply_events(
    events: Sequence[Mapping[str, Any]],
    applied: set[int],
    current_time: float,
    loads: dict[str, complex],
    devices: Mapping[str, RMSDevice],
) -> list[str]:
    labels: list[str] = []
    for index, event in enumerate(events):
        if index in applied or current_time + 1e-12 < float(event["time_s"]):
            continue
        event_type, target = str(event["type"]), str(event["device_id"])
        if event_type == "load_scale":
            if target not in loads:
                raise ValueError(f"Load-step target {target!r} is not a base load.")
            value = loads[target]
            loads[target] = complex(value.real * float(event.get("p_multiplier", 1.0)), value.imag * float(event.get("q_multiplier", 1.0)))
        elif event_type == "der_trip":
            if target not in devices:
                raise ValueError(f"DER-trip target {target!r} has no configured RMS model.")
            devices[target].trip()
        else:
            raise ValueError(f"Unsupported RMS disturbance type {event_type!r}.")
        applied.add(index)
        labels.append(f"{event_type}:{target}")
    return labels


def _result_rows(time_s: float, result: PFResult, device_states: Mapping[str, Mapping[str, float]], event: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    voltage_rows = [{"time_s": time_s, **record} for record in result.bus_voltage_records]
    summary: dict[str, Any] = {"time_s": time_s, "event": event, "pf_iterations": result.iterations, **result.summary}
    for device_id, state in device_states.items():
        summary[f"{device_id.lower()}_p_kw"] = float(state["p_kw"])
        summary[f"{device_id.lower()}_q_kvar"] = float(state["q_kvar"])
    return voltage_rows, summary


def run_rms(
    solver: PowerFlowSolver,
    network: NetworkModel,
    operating_point: DynamicOperatingPoint,
    rms_config: Mapping[str, Any],
    scenario_name: str,
) -> RMSResult:
    """Run seconds-scale first-order DER dynamics with algebraic network solves."""
    try:
        scenario = rms_config["scenarios"][scenario_name]
    except KeyError as error:
        available = ", ".join(rms_config["scenarios"])
        raise ValueError(f"Unknown RMS scenario {scenario_name!r}; available: {available}") from error
    dt, t_end = float(scenario["dt_s"]), float(scenario["t_end_s"])
    if dt <= 0 or t_end <= 0:
        raise ValueError("RMS scenario dt_s and t_end_s must be positive.")
    state, devices = initialize_rms(operating_point, rms_config)
    loads = dict(operating_point.load_pq)
    commands = dict(operating_point.der_pq)
    current_result = PFResult(True, state.bus_voltages, tuple(), tuple(), {"converged": True}, 0, state.pf_state or PFState())
    voltage_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    # Solve at t=0 too: this checks that the persisted QSTS operating point can be re-applied.
    initial_op = OperatingPoint(operating_point.time_index, loads, commands, {"rms_time_s": 0.0})
    current_result = solver.solve(initial_op, state.pf_state, SolverContext(time_step_hours=dt / 3600.0))
    state = RMSState(0.0, current_result.bus_voltages, state.device_states, current_result.next_state)
    rows, summary = _result_rows(0.0, current_result, state.device_states, "")
    voltage_rows.extend(rows); summary_rows.append(summary)

    events = list(scenario.get("events", []))
    applied: set[int] = set()
    for step in range(1, ceil(t_end / dt) + 1):
        time_s = min(step * dt, t_end)
        event_labels = _apply_events(events, applied, time_s, loads, devices)
        new_states: dict[str, dict[str, float]] = {}
        for device_id, device in devices.items():
            old_state = state.device_states[device_id]
            derivative = device.derivative(old_state)
            new_states[device_id] = {key: float(old_state[key]) + (time_s - state.time) * value for key, value in derivative.items()}
            commands[device_id] = device.injection(new_states[device_id], operating_point.der_pq[device_id])
        pf_op = OperatingPoint(operating_point.time_index, loads, commands, {"rms_time_s": time_s, "scenario": scenario_name})
        current_result = solver.solve(pf_op, state.pf_state, SolverContext(time_step_hours=(time_s - state.time) / 3600.0, previous_result=current_result))
        state = RMSState(time_s, current_result.bus_voltages, new_states, current_result.next_state)
        rows, summary = _result_rows(time_s, current_result, new_states, ";".join(event_labels))
        voltage_rows.extend(rows); summary_rows.append(summary)
    return RMSResult(pd.DataFrame(summary_rows), pd.DataFrame(voltage_rows), state)
