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
    """A controller-adjustable DER command for one QSTS operating point."""

    p_kw: float
    q_kvar: float
    controller_type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def with_pq(self, p_kw: float, q_kvar: float) -> "DERCommand":
        return DERCommand(p_kw, q_kvar, self.controller_type, self.parameters)


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
    """Result of one converged network solve."""

    converged: bool
    bus_voltages: Mapping[str, complex]
    bus_voltage_records: tuple[Mapping[str, Any], ...]
    branch_records: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, float | bool]
    iterations: int
    next_state: PFState
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
