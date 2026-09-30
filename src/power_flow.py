"""Solver-neutral network, operating-point, state, and result interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import pandas as pd


@dataclass(frozen=True)
class NetworkModel:
    """Immutable fixed network data shared by every power-flow backend."""

    base: Mapping[str, Any]
    buses: tuple[Mapping[str, Any], ...]
    branches: tuple[Mapping[str, Any], ...]
    devices: tuple[Mapping[str, Any], ...]
    simulation: Mapping[str, Any] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)
    line_codes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    shunt_devices: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    constraints: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    bus_ids: tuple[str, ...] = field(init=False)
    device_ids: tuple[str, ...] = field(init=False)
    branch_ids: tuple[str, ...] = field(init=False)
    shunt_ids: tuple[str, ...] = field(init=False)
    bus_index: Mapping[str, int] = field(init=False)
    device_index: Mapping[str, int] = field(init=False)
    branch_index: Mapping[str, int] = field(init=False)
    shunt_index: Mapping[str, int] = field(init=False)
    load_indices: tuple[int, ...] = field(init=False)
    der_indices: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        bus_ids = tuple(str(bus["id"]) for bus in self.buses)
        device_ids = tuple(str(device["id"]) for device in self.devices)
        branch_ids = tuple(str(branch["id"]) for branch in self.branches)
        shunt_ids = tuple(str(device["id"]) for device in self.shunt_devices)
        bus_index = {name.lower(): index for index, name in enumerate(bus_ids)}
        device_index = {name.lower(): index for index, name in enumerate(device_ids)}
        branch_index = {name.lower(): index for index, name in enumerate(branch_ids)}
        shunt_index = {name.lower(): index for index, name in enumerate(shunt_ids)}
        if len(bus_index) != len(bus_ids):
            raise ValueError("network.json has duplicate bus IDs.")
        if len(device_index) != len(device_ids):
            raise ValueError("network.json has duplicate device IDs.")
        if len(branch_index) != len(branch_ids) or len(shunt_index) != len(shunt_ids):
            raise ValueError("network.json has duplicate branch or shunt IDs.")
        object.__setattr__(self, "bus_ids", bus_ids)
        object.__setattr__(self, "device_ids", device_ids)
        object.__setattr__(self, "branch_ids", branch_ids)
        object.__setattr__(self, "shunt_ids", shunt_ids)
        object.__setattr__(self, "bus_index", MappingProxyType(bus_index))
        object.__setattr__(self, "device_index", MappingProxyType(device_index))
        object.__setattr__(self, "branch_index", MappingProxyType(branch_index))
        object.__setattr__(self, "shunt_index", MappingProxyType(shunt_index))
        object.__setattr__(self, "load_indices", tuple(index for index, device in enumerate(self.devices) if device["kind"] == "load"))
        object.__setattr__(self, "der_indices", tuple(index for index, device in enumerate(self.devices) if device["kind"] != "load"))

    @classmethod
    def from_json(cls, path: str | Path) -> "NetworkModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            base=raw["base"],
            buses=tuple(raw["buses"]),
            branches=tuple(raw["branches"]),
            devices=tuple(raw["devices"]),
            simulation=raw.get("simulation", {}),
            source=raw.get("source", {}),
            line_codes=raw.get("line_codes", {}),
            shunt_devices=tuple(raw.get("shunt_devices", ())),
            constraints=raw.get("constraints", {}),
            schema_version=str(raw.get("schema_version", "1.0")),
        )

    def device(self, name: str) -> Mapping[str, Any]:
        try:
            return self.devices[self.device_index[name.lower()]]
        except KeyError as exc:
            raise KeyError(f"Network has no device named {name!r}.") from exc

    def base_load_names(self) -> set[str]:
        return {self.device_ids[index] for index in self.load_indices}

    def simulation_timestamps(self) -> list[pd.Timestamp]:
        """Return the authoritative timeline from the ``simulation`` block.

        The timeline is ``[start, start+step, ..., end)`` using a half-open interval:
        ``end_datetime`` is the exclusive upper bound. A case must define all of
        ``start_datetime``, ``end_datetime`` and ``step_seconds``.
        """
        simulation = self.simulation
        required = {"start_datetime", "end_datetime", "step_seconds"}
        missing = required.difference(simulation)
        if missing:
            raise ValueError(f"network.json 'simulation' is missing fields: {sorted(missing)}")
        start = pd.Timestamp(simulation["start_datetime"])
        end = pd.Timestamp(simulation["end_datetime"])
        step_seconds = int(simulation["step_seconds"])
        if step_seconds <= 0:
            raise ValueError("network.json 'simulation.step_seconds' must be positive.")
        if end <= start:
            raise ValueError("network.json 'simulation.end_datetime' must be after 'start_datetime'.")
        timestamps = list(pd.date_range(start, end, freq=f"{step_seconds}s", inclusive="left"))
        if not timestamps:
            raise ValueError("network.json 'simulation' block yields an empty timeline.")
        return timestamps

    def step_seconds(self) -> int:
        simulation = self.simulation
        if "step_seconds" not in simulation:
            raise ValueError("network.json 'simulation' is missing 'step_seconds'.")
        step_seconds = int(simulation["step_seconds"])
        if step_seconds <= 0:
            raise ValueError("network.json 'simulation.step_seconds' must be positive.")
        return step_seconds


@dataclass(frozen=True)
class DERCommand:
    """A DER command plus its algebraic control law for one operating point.

    ``p_kw`` / ``q_kvar`` are the current setpoints. ``controller_type`` together
    with ``parameters`` describe how the setpoint responds to the terminal voltage;
    the network and controller equations are then solved jointly by the
    ``PowerFlowSolver`` backend.
    """

    p_kw: float
    q_kvar: float
    controller_type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def with_pq(self, p_kw: float, q_kvar: float) -> "DERCommand":
        return DERCommand(p_kw, q_kvar, self.controller_type, self.parameters)

    def with_q(self, q_kvar: float) -> "DERCommand":
        return DERCommand(self.p_kw, q_kvar, self.controller_type, self.parameters)

    def adjusted_q(self, voltage_pu: float) -> float:
        """Return the reactive-power command consistent with ``voltage_pu``.

        For ``constant_pq`` and ``soc_schedule`` the command is fixed. For
        ``volt_var`` it evaluates ``Q = clip[k*(Vref - V), -Qmax, Qmax]`` where the
        limit, reference and droop come from ``parameters``.
        """
        if self.controller_type != "volt_var":
            return self.q_kvar
        v_ref = float(self.parameters.get("v_ref_pu", 1.0))
        droop = float(self.parameters.get("droop_kvar_per_pu", 0.0))
        q_limit = float(self.parameters.get("q_limit_kvar", 0.0))
        q = droop * (v_ref - voltage_pu)
        return max(-q_limit, min(q_limit, q))


@dataclass(frozen=True)
class OperatingPoint:
    """All exogenous electrical injections for one time step.

    ``load_pq_kva`` uses positive consumption. DER commands use each model's declared
    sign convention; OpenDSSSolver maps them to their OpenDSS element type.
    """

    time: datetime
    load_pq_kva: Mapping[str, complex]
    der_commands: Mapping[str, DERCommand]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PFState:
    """Backend-neutral warm-start state and optional backend-specific payload."""

    voltage_guess: Mapping[str, complex] = field(default_factory=dict)
    backend_state: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstraintResult:
    """Per-operating-point security checks independent of a solver backend."""

    voltage_low_count: int = 0
    voltage_high_count: int = 0
    voltage_unbalance_count: int = 0
    line_thermal_violation_count: int = 0
    transformer_thermal_violation_count: int = 0
    min_voltage_pu: float = float("nan")
    max_voltage_pu: float = float("nan")
    max_voltage_unbalance_pct: float = float("nan")
    max_line_loading_pct: float = float("nan")
    max_transformer_loading_pct: float = float("nan")
    reverse_power_flow: bool = False
    reverse_power_kw: float = 0.0

    def summary(self) -> Mapping[str, float | bool]:
        return {
            "voltage_low_count": self.voltage_low_count,
            "voltage_high_count": self.voltage_high_count,
            "voltage_unbalance_count": self.voltage_unbalance_count,
            "line_thermal_violation_count": self.line_thermal_violation_count,
            "transformer_thermal_violation_count": self.transformer_thermal_violation_count,
            "max_voltage_unbalance_pct": self.max_voltage_unbalance_pct,
            "max_line_loading_pct": self.max_line_loading_pct,
            "max_transformer_loading_pct": self.max_transformer_loading_pct,
            "reverse_power_flow": self.reverse_power_flow,
            "reverse_power_kw": self.reverse_power_kw,
        }


@dataclass(frozen=True)
class HostingCapacityMetrics:
    """Observed operating headroom; not a separate DER-capacity optimization."""

    voltage_headroom_pu: float = float("nan")
    line_loading_headroom_pct: float = float("nan")
    transformer_loading_headroom_pct: float = float("nan")
    binding_constraint: str = "none"

    def summary(self) -> Mapping[str, float | str]:
        return {
            "voltage_headroom_pu": self.voltage_headroom_pu,
            "line_loading_headroom_pct": self.line_loading_headroom_pct,
            "transformer_loading_headroom_pct": self.transformer_loading_headroom_pct,
            "binding_constraint": self.binding_constraint,
        }


@dataclass(frozen=True)
class PFResult:
    """Result of one converged network solve.

    ``final_der_commands`` holds the DER commands after the network and DER control
    equations were solved jointly; the backend may have adjusted them from the
    original ``OperatingPoint`` setpoints. ``control_iterations`` counts the inner
    control updates used to reach that joint solution.
    """

    converged: bool
    bus_voltages: Mapping[str, complex]
    bus_voltage_records: tuple[Mapping[str, Any], ...]
    branch_records: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, float | bool]
    iterations: int
    next_state: PFState
    final_der_commands: Mapping[str, DERCommand] = field(default_factory=dict)
    control_iterations: int = 0
    pf_solve_count: int = 0
    pf_iterations_total: int = 0
    pf_iterations_last: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    constraints: ConstraintResult = field(default_factory=ConstraintResult)
    hosting_capacity: HostingCapacityMetrics = field(default_factory=HostingCapacityMetrics)


@dataclass(frozen=True)
class SolverContext:
    """Extension point for future coupled equations, scenarios, and solver options."""

    time_step_hours: float = 1.0
    control_iteration: int = 0
    previous_result: PFResult | None = None
    max_control_iterations: int = 20
    control_tolerance: float = 1e-6
    relaxation: float = 1.0
    extra_equations: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)


class PowerFlowSolver(Protocol):
    """Contract implemented by OpenDSS and future custom PF solvers."""

    def build(self, network: NetworkModel) -> None:
        """Build/compile immutable network data."""

    def solve(
        self,
        operating_point: OperatingPoint,
        state: PFState | None,
        context: SolverContext | None = None,
    ) -> PFResult:
        """Solve one operating point and return a state for the next time step."""
