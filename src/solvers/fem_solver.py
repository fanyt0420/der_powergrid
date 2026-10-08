"""P1 graph finite-element AC feeder prototype.

On each line the phasor field is piecewise linear.  The weak-form element
stiffness is (Z_per_length * length)**-1 [[1, -1], [-1, 1]].  For this
constant-coefficient, unshunted model it equals a series-only nodal Ybus;
this is a correctness experiment, not a claim of faster computation.

Scope: wye PQ loads and wye generator/EV DER on a three-phase, line-only
feeder with an ideal fixed source. Transformers, regulators, switches,
line charging, delta devices and neutral conductors are rejected at build.
"""

from __future__ import annotations

from cmath import phase
from math import degrees, sqrt
from typing import Any, Mapping

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from src.power_flow import DERCommand, NetworkModel, OperatingPoint, PFResult, PFState, SolverContext


_PHASE_ANGLE = {1: 0.0, 2: -2.0 * np.pi / 3.0, 3: 2.0 * np.pi / 3.0}


class FiniteElementPFSolver:
    """Solve line-only three-phase feeder snapshots through P1 weak-form assembly."""

    def __init__(self, *, voltage_tolerance_pu: float = 1e-9, max_iterations: int = 100) -> None:
        self.voltage_tolerance_pu = voltage_tolerance_pu
        self.max_iterations = max_iterations
        self.network: NetworkModel | None = None

    def build(self, network: NetworkModel) -> None:
        if network.shunt_devices:
            raise NotImplementedError("FEM prototype does not model shunt devices.")
        if any(str(branch["kind"]).lower() != "line" for branch in network.branches):
            raise NotImplementedError("FEM prototype supports line branches only; transformers, regulators and switches require new element forms.")
        if any(str(device.get("connection", "wye")).lower() != "wye" for device in network.devices):
            raise NotImplementedError("FEM prototype supports wye-connected devices only.")
        if any(str(device["kind"]).lower() not in {"load", "pv", "wind", "ev"} for device in network.devices):
            raise NotImplementedError("FEM prototype supports loads, PV, wind and EV; storage and other devices require a separate electrical model.")
        if network.source and any(key in network.source for key in ("mvasc3", "mvasc1", "r1", "x1", "r0", "x0")):
            raise NotImplementedError("FEM prototype assumes an ideal source; source impedance is not supported.")
        for branch in network.branches:
            if tuple(branch["phases"]) != tuple(branch.get("from_terminal_nodes", branch["phases"])) or tuple(branch["phases"]) != tuple(branch.get("to_terminal_nodes", branch["phases"])):
                raise NotImplementedError("FEM prototype requires identical phase numbering at both line ends.")

        keys = tuple((str(bus["id"]), int(p)) for bus in network.buses for p in bus["phases"])
        index = {(bus.lower(), p): i for i, (bus, p) in enumerate(keys)}
        if len(index) != len(keys):
            raise ValueError("Duplicate bus-phase node.")
        nominal_v = np.asarray([float(bus["nominal_kv_ll"]) * 1000.0 / sqrt(3.0) for bus in network.buses for _ in bus["phases"]])
        slack_name = str(network.base["slack_bus"]).lower()
        source_indices = np.asarray([i for i, (bus, _) in enumerate(keys) if bus.lower() == slack_name], dtype=int)
        if len(source_indices) != 3:
            raise NotImplementedError("FEM prototype requires a three-phase fixed source.")
        source_pu = float(network.source.get("voltage_pu", 1.0))
        source_angle = np.deg2rad(float(network.source.get("angle_deg", 0.0)))
        fixed_v = nominal_v[source_indices] * source_pu * np.exp(1j * np.asarray([_PHASE_ANGLE[keys[i][1]] + source_angle for i in source_indices]))

        rows: list[int] = []
        cols: list[int] = []
        values: list[complex] = []
        elements: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
        for branch in network.branches:
            phases = tuple(int(p) for p in branch["phases"])
            from_bus, to_bus = str(branch["from_bus"]), str(branch["to_bus"])
            left = np.asarray([index[(from_bus.lower(), p)] for p in phases], dtype=int)
            right = np.asarray([index[(to_bus.lower(), p)] for p in phases], dtype=int)
            length = float(branch.get("length", branch.get("length_km")))
            if "line_code" in branch:
                code = network.line_codes[str(branch["line_code"])]
                if "c_nf_per_unit" in code and np.any(np.asarray(code["c_nf_per_unit"], dtype=float) != 0):
                    raise NotImplementedError("FEM prototype does not model line charging capacitance.")
                unit = str(code.get("unit", "km")).lower()
                if unit not in {"km", "mi"}:
                    raise NotImplementedError(f"Unsupported line-code unit: {unit}")
                z_per_unit = np.asarray(code["r_ohm_per_unit"], dtype=float) + 1j * np.asarray(code["x_ohm_per_unit"], dtype=float)
                z = z_per_unit * length
            else:
                z = (np.asarray([[complex(*entry) for entry in row] for row in branch["z_ohm_per_km"]]) * length)
            if z.shape != (len(phases), len(phases)):
                raise ValueError(f"Impedance dimensions do not match {branch['id']}.")
            try:
                conductance = np.linalg.inv(z)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"Singular series impedance in {branch['id']}.") from exc
            # Integral B.T @ Z_per_length^-1 @ B over this P1 line element.
            for a in range(len(phases)):
                for b in range(len(phases)):
                    y = conductance[a, b]
                    for i, j, sign in ((left[a], left[b], 1), (right[a], right[b], 1), (left[a], right[b], -1), (right[a], left[b], -1)):
                        rows.append(int(i)); cols.append(int(j)); values.append(sign * y)
            elements.append((branch, left, right, z))
        ybus = coo_matrix((np.asarray(values), (rows, cols)), shape=(len(keys), len(keys))).tocsc()
        free = np.asarray([i for i in range(len(keys)) if i not in set(source_indices)], dtype=int)
        try:
            lu = splu(ybus[free, :][:, free].tocsc())
        except RuntimeError as exc:
            raise ValueError("FEM matrix is singular; check feeder connectivity and source.") from exc
        self.network = network
        self.keys, self.index, self.nominal_v = keys, index, nominal_v
        self.source_indices, self.fixed_v, self.free = source_indices, fixed_v, free
        self.ybus, self.elements, self.lu = ybus, elements, lu
        self.source_rhs = ybus[free, :][:, source_indices] @ fixed_v

    def _injections(self, point: OperatingPoint, commands: Mapping[str, DERCommand]) -> np.ndarray:
        assert self.network is not None
        expected_loads = {str(self.network.devices[i]["id"]).lower() for i in self.network.load_indices}
        expected_der = {str(self.network.devices[i]["id"]).lower() for i in self.network.der_indices}
        if {name.lower() for name in point.load_pq_kva} != expected_loads or {name.lower() for name in commands} != expected_der:
            raise ValueError("Operating point must contain exactly every load and DER.")
        load_by_key = {name.lower(): complex(power) for name, power in point.load_pq_kva.items()}
        command_by_key = {name.lower(): command for name, command in commands.items()}
        power = np.zeros(len(self.keys), dtype=complex)
        for device in self.network.devices:
            kind = str(device["kind"]).lower()
            name = str(device["id"]).lower()
            if kind == "load":
                s = -load_by_key[name]
            else:
                command = command_by_key[name]
                if kind == "ev":
                    s = -complex(command.p_kw, command.q_kvar)
                else:
                    s = complex(command.p_kw, command.q_kvar)
            phases = tuple(int(p) for p in device.get("terminal_nodes", device["phases"]))
            for p in phases:
                power[self.index[(str(device["bus"]).lower(), p)]] += s * 1000.0 / len(phases)
        return power

    def _solve_network(self, point: OperatingPoint, commands: Mapping[str, DERCommand], initial: np.ndarray) -> tuple[np.ndarray, int]:
        power = self._injections(point, commands)
        v = initial.copy()
        v[self.source_indices] = self.fixed_v
        for iteration in range(1, self.max_iterations + 1):
            if np.any(np.abs(v[self.free]) < 0.2 * self.nominal_v[self.free]):
                raise RuntimeError("FEM voltage fell below 0.2 pu; fixed-point model is unsafe.")
            rhs = np.conj(power[self.free] / v[self.free]) - self.source_rhs
            updated = self.lu.solve(rhs)
            delta = float(np.max(np.abs(updated - v[self.free]) / self.nominal_v[self.free]))
            v[self.free] = updated
            if delta < self.voltage_tolerance_pu:
                residual = self.ybus[self.free, :] @ v - np.conj(power[self.free] / v[self.free])
                if np.max(np.abs(residual)) > 1e-6 * max(1.0, float(np.max(np.abs(power[self.free] / v[self.free])))):
                    raise RuntimeError("FEM nonlinear KCL residual failed.")
                return v, iteration
        raise RuntimeError(f"FEM current-injection iteration did not converge at {point.time}.")

    def solve(self, operating_point: OperatingPoint, state: PFState | None, context: SolverContext | None = None) -> PFResult:
        if self.network is None:
            raise RuntimeError("Call build(network) before solve().")
        context = context or SolverContext()
        initial = self.nominal_v * np.exp(1j * np.asarray([_PHASE_ANGLE[p] for _, p in self.keys]))
        if state is not None:
            for i, (bus, p) in enumerate(self.keys):
                guess = state.voltage_guess.get(f"{bus}.{p}")
                if guess is not None:
                    initial[i] = complex(guess) * self.nominal_v[i]
        commands = dict(operating_point.der_commands)
        solves = 0
        total_iterations = 0
        control_iterations = 0
        while True:
            v, iterations = self._solve_network(operating_point, commands, initial)
            solves += 1
            total_iterations += iterations
            changes: dict[str, DERCommand] = {}
            for name, command in commands.items():
                if command.controller_type != "volt_var":
                    continue
                device = self.network.device(name)
                phases = tuple(int(p) for p in command.parameters.get("terminal_nodes", device["phases"]))
                measured = float(np.mean([abs(v[self.index[(str(device["bus"]).lower(), p)]]) / self.nominal_v[self.index[(str(device["bus"]).lower(), p)]] for p in phases]))
                q = command.q_kvar + context.relaxation * (command.adjusted_q(measured) - command.q_kvar)
                if abs(q - command.q_kvar) > context.control_tolerance:
                    changes[name] = command.with_q(q)
            if not changes:
                break
            control_iterations += 1
            if control_iterations > context.max_control_iterations:
                raise RuntimeError(f"FEM DER controls did not converge at {operating_point.time}.")
            commands.update(changes)
            initial = v

        pu = v / self.nominal_v
        voltage_map = {f"{bus}.{p}": complex(pu[i]) for i, (bus, p) in enumerate(self.keys)}
        voltage_records = tuple({"bus": bus, "node": p, "v_pu": float(abs(pu[i])), "angle_deg": float(degrees(phase(pu[i])))} for i, (bus, p) in enumerate(self.keys))
        branch_records: list[dict[str, Any]] = []
        losses = 0j
        for branch, left, right, z in self.elements:
            current = np.linalg.solve(z, v[left] - v[right])
            powers = v[left] * np.conj(current) / 1000.0
            losses += np.sum((v[left] - v[right]) * np.conj(current)) / 1000.0
            branch_records.append({
                "branch": str(branch["id"]), "branch_kind": "line", "bus1": str(branch["from_bus"]), "bus2": str(branch["to_bus"]),
                "max_current_a": float(np.max(np.abs(current))), "tap_pu": float("nan"), "tap_position": float("nan"),
                "terminal1_p_kw": float(np.sum(powers.real)), "terminal1_q_kvar": float(np.sum(powers.imag)),
                "terminal1_apparent_kva": float(np.sum(np.abs(powers))),
            })
        source_current = self.ybus[self.source_indices, :] @ v
        source_power = np.sum(v[self.source_indices] * np.conj(source_current)) / 1000.0
        magnitudes = np.abs(pu)
        summary: dict[str, float | bool] = {
            "converged": True, "min_v_pu": float(np.min(magnitudes)), "max_v_pu": float(np.max(magnitudes)),
            "loss_kw": float(losses.real), "loss_kvar": float(losses.imag),
            "source_p_kw": float(-source_power.real), "source_q_kvar": float(-source_power.imag),
        }
        return PFResult(
            converged=True, bus_voltages=voltage_map, bus_voltage_records=voltage_records,
            branch_records=tuple(branch_records), summary=summary, iterations=iterations,
            next_state=PFState(voltage_guess=voltage_map), final_der_commands=commands,
            control_iterations=control_iterations, pf_solve_count=solves,
            pf_iterations_total=total_iterations, pf_iterations_last=iterations,
            metadata={"backend": "p1_graph_fem", "model": "series-only, ideal source, wye PQ"},
        )
