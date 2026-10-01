"""Solver-neutral network, operating-point, state, and result interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from math import isfinite, sqrt
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
        self._validate()

    @staticmethod
    def _phases(value: Any, location: str) -> tuple[int, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"{location}.phases must be a non-empty list drawn from [1, 2, 3].")
        phases = tuple(value)
        if any(type(phase) is not int or phase not in {1, 2, 3} for phase in phases) or len(set(phases)) != len(phases):
            raise ValueError(f"{location}.phases must contain unique integer phases drawn from [1, 2, 3].")
        return phases

    @staticmethod
    def _positive(value: Any, location: str, *, allow_zero: bool = False) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{location} must be a finite {'non-negative' if allow_zero else 'positive'} number.") from exc
        if not isfinite(number) or number < 0 or (not allow_zero and number == 0):
            raise ValueError(f"{location} must be a finite {'non-negative' if allow_zero else 'positive'} number.")
        return number

    @staticmethod
    def _matrix(matrix: Any, size: int, location: str, *, nonnegative: bool = True) -> None:
        if not isinstance(matrix, (list, tuple)) or len(matrix) != size:
            raise ValueError(f"{location} must be a {size} x {size} matrix matching its phases.")
        for row in matrix:
            if not isinstance(row, (list, tuple)) or len(row) != size:
                raise ValueError(f"{location} must be a {size} x {size} matrix matching its phases.")
            for value in row:
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{location} entries must be finite numbers.") from exc
                if not isfinite(number) or (nonnegative and number < 0):
                    raise ValueError(f"{location} entries must be finite {'non-negative ' if nonnegative else ''}numbers.")

    def _bus(self, bus_id: Any, location: str) -> Mapping[str, Any]:
        try:
            return self.buses[self.bus_index[str(bus_id).lower()]]
        except KeyError as exc:
            raise ValueError(f"{location} references unknown bus {bus_id!r}.") from exc

    def _terminal_phases(self, item: Mapping[str, Any], bus: Mapping[str, Any], location: str, phase_key: str = "terminal_nodes") -> tuple[int, ...]:
        phases = self._phases(item.get(phase_key, item.get("phases")), location)
        bus_phases = self._phases(bus.get("phases"), f"bus {bus['id']}")
        if not set(phases).issubset(bus_phases):
            raise ValueError(f"{location}.phases {list(phases)} are not present on bus {bus['id']!r}.")
        return phases

    def _validate_voltage(self, item: Mapping[str, Any], bus: Mapping[str, Any], phases: tuple[int, ...], location: str) -> None:
        key = "kv" if "kv" in item else "kv_ll" if "kv_ll" in item else "kv_ln" if "kv_ln" in item else None
        if key is None:
            return
        actual = self._positive(item[key], f"{location}.{key}")
        if "nominal_kv_ll" not in bus:
            return
        nominal_ll = self._positive(bus["nominal_kv_ll"], f"bus {bus['id']}.nominal_kv_ll")
        connection = str(item.get("connection", "wye")).lower()
        expected = nominal_ll / sqrt(3.0) if key == "kv_ln" or (len(phases) == 1 and connection == "wye") else nominal_ll
        if abs(actual - expected) > max(0.02, 0.02 * expected):
            raise ValueError(f"{location}.{key}={actual:g} is inconsistent with bus {bus['id']!r} nominal voltage ({expected:g} kV expected).")

    def _validate(self) -> None:
        """Reject ambiguous or electrically inconsistent fixed-network input early."""
        self._positive(self.base.get("frequency_hz"), "base.frequency_hz")
        self._positive(self.base.get("base_kv_ll"), "base.base_kv_ll")
        slack = self._bus(self.base.get("slack_bus"), "base.slack_bus")
        for bus in self.buses:
            location = f"bus {bus['id']!r}"
            self._phases(bus.get("phases"), location)
            self._positive(bus.get("nominal_kv_ll"), f"{location}.nominal_kv_ll")
        if not bool(slack.get("is_slack", False)):
            raise ValueError("base.slack_bus must refer to a bus with is_slack=true.")
        if sum(bool(bus.get("is_slack", False)) for bus in self.buses) != 1:
            raise ValueError("network.json must define exactly one slack bus.")

        if self.source:
            source_bus = self._bus(self.source.get("bus"), "source.bus")
            source_phases = self._terminal_phases(self.source, source_bus, "source")
            if source_bus["id"].lower() != slack["id"].lower():
                raise ValueError("source.bus must match base.slack_bus.")
            self._positive(self.source.get("base_kv_ll"), "source.base_kv_ll")
            for rating in ("mvasc3", "mvasc1"):
                if rating in self.source:
                    self._positive(self.source[rating], f"source.{rating}")
            self._validate_voltage({"kv_ll": self.source["base_kv_ll"], "connection": "delta"}, source_bus, source_phases, "source")

        for name, code in self.line_codes.items():
            location = f"line_codes.{name!r}"
            raw_matrix = code.get("r_ohm_per_unit")
            inferred_phases = tuple(range(1, len(raw_matrix) + 1)) if isinstance(raw_matrix, (list, tuple)) else ()
            phases = self._phases(code.get("phases", inferred_phases), location)
            self._matrix(code.get("r_ohm_per_unit"), len(phases), f"{location}.r_ohm_per_unit")
            self._matrix(code.get("x_ohm_per_unit"), len(phases), f"{location}.x_ohm_per_unit", nonnegative=False)

        for branch in self.branches:
            location = f"branch {branch['id']!r}"
            kind = str(branch.get("kind", "")).lower()
            if kind not in {"line", "switch", "transformer", "regulator"}:
                raise ValueError(f"{location}.kind must be line, switch, transformer, or regulator.")
            from_key, to_key = ("from_bus", "to_bus") if "from_bus" in branch or "to_bus" in branch else ("from", "to")
            bus_from = self._bus(branch.get(from_key), f"{location}.{from_key}")
            bus_to = self._bus(branch.get(to_key), f"{location}.{to_key}")
            phases = self._phases(branch.get("phases"), location)
            from_phases = self._terminal_phases(branch, bus_from, f"{location}.from_terminal_nodes", "from_terminal_nodes")
            to_phases = self._terminal_phases(branch, bus_to, f"{location}.to_terminal_nodes", "to_terminal_nodes")
            if set(from_phases) != set(phases) or set(to_phases) != set(phases):
                raise ValueError(f"{location} endpoint terminal nodes must match branch.phases.")
            if kind in {"line", "switch"}:
                if "normal_amps" in branch:
                    self._positive(branch["normal_amps"], f"{location}.normal_amps")
                if kind == "line":
                    self._positive(branch.get("length", branch.get("length_km")), f"{location}.length")
                    if "line_code" in branch:
                        try:
                            code = self.line_codes[str(branch["line_code"])]
                        except KeyError as exc:
                            raise ValueError(f"{location}.line_code references unknown code {branch['line_code']!r}.") from exc
                        code_matrix = code.get("r_ohm_per_unit")
                        inferred = tuple(range(1, len(code_matrix) + 1)) if isinstance(code_matrix, (list, tuple)) else ()
                        if len(self._phases(code.get("phases", inferred), f"line_codes.{branch['line_code']!r}")) != len(phases):
                            raise ValueError(f"{location}.phases and line_codes.{branch['line_code']!r}.phases have different dimensions.")
                    elif "z_ohm_per_km" in branch:
                        impedance = branch["z_ohm_per_km"]
                        if not isinstance(impedance, (list, tuple)) or len(impedance) != len(phases):
                            raise ValueError(f"{location}.z_ohm_per_km must match the branch phase dimension.")
                        for row in impedance:
                            if not isinstance(row, (list, tuple)) or len(row) != len(phases):
                                raise ValueError(f"{location}.z_ohm_per_km must match the branch phase dimension.")
                    else:
                        raise ValueError(f"{location} must define line_code or z_ohm_per_km.")
            else:
                windings = branch.get("windings")
                if not isinstance(windings, (list, tuple)) or len(windings) < 2:
                    raise ValueError(f"{location}.windings must define at least two complete windings.")
                winding_buses: list[str] = []
                for index, winding in enumerate(windings, start=1):
                    winding_location = f"{location}.windings[{index}]"
                    winding_bus = self._bus(winding.get("bus"), f"{winding_location}.bus")
                    winding_phases = self._terminal_phases(winding, winding_bus, winding_location)
                    connection = str(winding.get("connection", "")).lower()
                    if connection not in {"wye", "delta"}:
                        raise ValueError(f"{winding_location}.connection must be wye or delta.")
                    voltage_key = "kv_ll" if "kv_ll" in winding else "kv_ln" if "kv_ln" in winding else None
                    if voltage_key is None:
                        raise ValueError(f"{winding_location} must define kv_ll or kv_ln.")
                    self._positive(winding[voltage_key], f"{winding_location}.{voltage_key}")
                    self._positive(winding.get("kva"), f"{winding_location}.kva")
                    self._validate_voltage({voltage_key: winding[voltage_key], "connection": connection}, winding_bus, winding_phases, winding_location)
                    winding_buses.append(str(winding_bus["id"]).lower())
                if winding_buses[:2] != [str(bus_from["id"]).lower(), str(bus_to["id"]).lower()]:
                    raise ValueError(f"{location}.from/to must match the first two transformer windings.")
                xhl_key = "xhl_pct" if "xhl_pct" in branch else "xhl_percent" if "xhl_percent" in branch else None
                if xhl_key is not None:
                    self._positive(branch[xhl_key], f"{location}.{xhl_key}")
                if "tap" in branch:
                    tap = branch["tap"]
                    minimum = self._positive(tap.get("min_pu"), f"{location}.tap.min_pu")
                    maximum = self._positive(tap.get("max_pu"), f"{location}.tap.max_pu")
                    if minimum >= maximum:
                        raise ValueError(f"{location}.tap.min_pu must be less than tap.max_pu.")
                    if int(tap.get("num_taps", 0)) <= 0:
                        raise ValueError(f"{location}.tap.num_taps must be a positive integer.")

        for device in (*self.devices, *self.shunt_devices):
            location = f"device {device['id']!r}"
            kind = str(device.get("kind", "")).lower()
            allowed = {"load", "pv", "wind", "bess", "ev", "capacitor"}
            if kind not in allowed:
                raise ValueError(f"{location}.kind must be one of {sorted(allowed)}.")
            bus = self._bus(device.get("bus"), f"{location}.bus")
            phases = self._terminal_phases(device, bus, location)
            connection = str(device.get("connection", "wye")).lower()
            if connection not in {"wye", "delta"}:
                raise ValueError(f"{location}.connection must be wye or delta.")
            self._validate_voltage(device, bus, phases, location)
            for rating in ("p_rated_kw", "q_rated_kvar", "s_rated_kva", "energy_kwh"):
                if rating in device:
                    self._positive(device[rating], f"{location}.{rating}")
            for injection in ("base_p_kw", "base_q_kvar", "q_kvar"):
                if injection in device:
                    self._positive(device[injection], f"{location}.{injection}", allow_zero=True)
            if kind != "load" and "p_rated_kw" not in device and kind != "capacitor":
                raise ValueError(f"{location} must define positive p_rated_kw.")
            if kind == "bess":
                initial_key = "initial_soc_pct" if "initial_soc_pct" in device else "initial_soc"
                reserve_key = "reserve_soc_pct" if "reserve_soc_pct" in device else "reserve_soc"
                for field_name in (initial_key, reserve_key):
                    value = self._positive(device.get(field_name), f"{location}.{field_name}", allow_zero=True)
                    if value > 100:
                        raise ValueError(f"{location}.{field_name} must be in [0, 100].")
                if float(device[initial_key]) < float(device[reserve_key]):
                    raise ValueError(f"{location}.{initial_key} must be no less than {reserve_key}.")

        for key in ("voltage_min_pu", "voltage_max_pu", "voltage_unbalance_limit_pct", "line_loading_limit_pct", "transformer_loading_limit_pct"):
            if key in self.constraints:
                self._positive(self.constraints[key], f"constraints.{key}")
        if "voltage_min_pu" in self.constraints and "voltage_max_pu" in self.constraints and float(self.constraints["voltage_min_pu"]) >= float(self.constraints["voltage_max_pu"]):
            raise ValueError("constraints.voltage_min_pu must be less than voltage_max_pu.")

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
    final_der_commands: Mapping[str, DERCommand]
    control_iterations: int = 0
    pf_solve_count: int = 0
    pf_iterations_total: int = 0
    pf_iterations_last: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
