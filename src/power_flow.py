"""Solver-neutral network, operating-point, state, and result interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class NetworkModel:
    """Immutable fixed network data shared by every power-flow backend."""

    base: Mapping[str, Any]
    buses: tuple[Mapping[str, Any], ...]
    branches: tuple[Mapping[str, Any], ...]
    devices: tuple[Mapping[str, Any], ...]
    simulation: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "NetworkModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            base=raw["base"],
            buses=tuple(raw["buses"]),
            branches=tuple(raw["branches"]),
            devices=tuple(raw["devices"]),
            simulation=raw.get("simulation", {}),
        )

    def device(self, name: str) -> Mapping[str, Any]:
        for device in self.devices:
            if str(device["id"]).lower() == name.lower():
                return device
        raise KeyError(f"Network has no device named {name!r}.")

    def base_load_names(self) -> set[str]:
        return {str(device["id"]) for device in self.devices if device["kind"] == "load"}


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
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverContext:
    """Extension point for future coupled equations, scenarios, and solver options."""

    time_step_hours: float = 1.0
    control_iteration: int = 0
    previous_result: PFResult | None = None
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
