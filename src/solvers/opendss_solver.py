"""OpenDSS implementation of the solver-neutral PowerFlowSolver interface."""

from __future__ import annotations

from cmath import rect
from math import radians
from pathlib import Path
from typing import Any, Mapping

from dss import dss
import numpy as np

from src.power_flow import DERCommand, NetworkModel, OperatingPoint, PFResult, PFState, SolverContext


class OpenDSSSolver:
    """Adapter that applies an OperatingPoint to OpenDSS and solves it."""

    def __init__(self, master_file: str | Path) -> None:
        self.master_file = Path(master_file).resolve()
        self.network: NetworkModel | None = None
        self._load_name_by_key: dict[str, str] = {}
        self._device_by_key: dict[str, Mapping[str, Any]] = {}
        self._branch_by_key: dict[str, Mapping[str, Any]] = {}

    def build(self, network: NetworkModel) -> None:
        if not self.master_file.exists():
            raise FileNotFoundError(f"OpenDSS master file not found: {self.master_file}")
        dss("Clear")
        dss(f'Redirect "{self.master_file}"')
        self.network = network
        self._load_name_by_key = {network.device_ids[index].lower(): network.device_ids[index] for index in network.load_indices}
        self._device_by_key = {name.lower(): network.devices[index] for name, index in network.device_index.items()}
        self._branch_by_key = {name.lower(): network.branches[index] for name, index in network.branch_index.items()}

    def _network(self) -> NetworkModel:
        if self.network is None:
            raise RuntimeError("Call solver.build(network) before solver.solve(...).")
        return self.network

    def _set_loads(self, load_pq_kva: Mapping[str, complex]) -> None:
        expected = self._load_name_by_key
        provided = {name.lower(): value for name, value in load_pq_kva.items()}
        unknown, missing = sorted(set(provided) - set(expected)), sorted(set(expected) - set(provided))
        if unknown or missing:
            raise ValueError(f"Invalid load operating point; unknown={unknown}, missing={missing}")
        loads = dss.ActiveCircuit.Loads
        for key, canonical_name in expected.items():
            value = complex(provided[key])
            loads.Name, loads.kW, loads.kvar = canonical_name, value.real, value.imag

    def _set_der_command(self, name: str, command: DERCommand) -> None:
        kind = str(self._device_by_key[name.lower()]["kind"])
        if kind in {"pv", "wind"}:
            elements = dss.ActiveCircuit.Generators
            elements.Name, elements.kW, elements.kvar = name, command.p_kw, command.q_kvar
        elif kind == "ev":
            elements = dss.ActiveCircuit.Loads
            elements.Name, elements.kW, elements.kvar = name, command.p_kw, command.q_kvar
        elif kind == "bess":
            state = "Charging" if command.p_kw < 0 else "Discharging" if command.p_kw > 0 else "Idling"
            # OpenDSS Storage uses the opposite active-power convention from a
            # Load: positive kW discharges into the grid and negative kW charges.
            # Keep the signed QSTS command; abs(p_kw) would turn a charge request
            # into a discharge request.
            dss(f"Storage.{name}.kW={command.p_kw}")
            dss(f"Storage.{name}.State={state}")
            dss(f"Storage.{name}.kvar={command.q_kvar}")
            if "soc_pct" in command.parameters:
                dss(f"Storage.{name}.%stored={float(command.parameters['soc_pct'])}")
        else:
            raise ValueError(f"No OpenDSS mapping for DER kind {kind!r}.")

    def _set_der_commands(self, der_commands: Mapping[str, DERCommand]) -> None:
        for name, command in der_commands.items():
            self._set_der_command(name, command)

    def solve(self, operating_point: OperatingPoint, state: PFState | None, context: SolverContext | None = None) -> PFResult:
        self._network()
        self._set_loads(operating_point.load_pq_kva)

        context = context or SolverContext()
        max_control_iterations = context.max_control_iterations
        tolerance = context.control_tolerance
        relaxation = context.relaxation

        commands: dict[str, DERCommand] = dict(operating_point.der_commands)

        solution = dss.ActiveCircuit.Solution
        pf_solve_count = 0
        pf_iterations_total = 0
        pf_iterations_last = 0

        def solve_once() -> None:
            nonlocal pf_solve_count, pf_iterations_total, pf_iterations_last
            # A control iteration is an algebraic fixed-point iteration, not a
            # physical time advance. Reapply *every* DER command before every
            # OpenDSS solve so stateful elements (especially Storage) are frozen
            # at the same QSTS operating point while Volt-VAR commands change.
            self._set_der_commands(commands)
            solution.Solve()
            if not solution.Converged:
                raise RuntimeError(f"OpenDSS power flow did not converge at timestamp {operating_point.time}.")
            iterations = int(solution.Iterations)
            pf_solve_count += 1
            pf_iterations_total += iterations
            pf_iterations_last = iterations

        solve_once()

        control_iterations = 0
        while True:
            voltages = self._bus_voltage_records()[1]
            active_names = [name for name, command in commands.items() if command.controller_type == "volt_var"]
            if not active_names:
                break
            voltage_pu = np.empty(len(active_names))
            for index, name in enumerate(active_names):
                command, device = commands[name], self._device_by_key[name.lower()]
                prefix = f"{device['bus'].lower()}."
                terminal_nodes = {int(node) for node in command.parameters.get("terminal_nodes", ())}
                values = [abs(value) for key, value in voltages.items() if key.lower().startswith(prefix) and (not terminal_nodes or int(key.rsplit(".", 1)[1]) in terminal_nodes)]
                if not values:
                    raise ValueError(f"Volt-VAR DER {name!r} found no bus voltage for bus {device['bus']!r} at timestamp {operating_point.time}.")
                voltage_pu[index] = np.mean(values)
            prior_q = np.fromiter((commands[name].q_kvar for name in active_names), dtype=float)
            v_ref = np.fromiter((float(commands[name].parameters.get("v_ref_pu", 1.0)) for name in active_names), dtype=float)
            droop = np.fromiter((float(commands[name].parameters.get("droop_kvar_per_pu", 0.0)) for name in active_names), dtype=float)
            limit = np.fromiter((float(commands[name].parameters.get("q_limit_kvar", 0.0)) for name in active_names), dtype=float)
            updated_q = prior_q + relaxation * (np.clip(droop * (v_ref - voltage_pu), -limit, limit) - prior_q)
            changed = bool(np.any(np.abs(updated_q - prior_q) > tolerance))
            if not changed:
                break
            control_iterations += 1
            if control_iterations > max_control_iterations:
                raise RuntimeError(f"DER controls did not converge at timestamp {operating_point.time}.")
            commands = {**commands, **{name: commands[name].with_q(float(q_kvar)) for name, q_kvar in zip(active_names, updated_q, strict=True)}}
            solve_once()

        voltage_records, voltages = self._bus_voltage_records()
        branch_records = self._branch_records()
        summary = self._system_summary(voltage_records)
        return PFResult(
            converged=True,
            bus_voltages=voltages,
            bus_voltage_records=tuple(voltage_records),
            branch_records=tuple(branch_records),
            summary=summary,
            iterations=pf_iterations_last,
            next_state=PFState(voltage_guess=voltages),
            final_der_commands=commands,
            control_iterations=control_iterations,
            pf_solve_count=pf_solve_count,
            pf_iterations_total=pf_iterations_total,
            pf_iterations_last=pf_iterations_last,
            metadata={"backend": "opendss", "context": context},
        )

    @staticmethod
    def _bus_voltage_records() -> tuple[list[dict[str, Any]], dict[str, complex]]:
        records: list[dict[str, Any]] = []
        voltages: dict[str, complex] = {}
        circuit = dss.ActiveCircuit
        for bus_name in circuit.AllBusNames:
            circuit.SetActiveBus(bus_name)
            for idx, node in enumerate(circuit.ActiveBus.Nodes):
                values = circuit.ActiveBus.puVmagAngle
                magnitude, angle = float(values[2 * idx]), float(values[2 * idx + 1])
                records.append({"bus": bus_name, "node": int(node), "v_pu": magnitude, "angle_deg": angle})
                voltages[f"{bus_name}.{node}"] = rect(magnitude, radians(angle))
        return records, voltages

    def _branch_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        lines, transformers, circuit = dss.ActiveCircuit.Lines, dss.ActiveCircuit.Transformers, dss.ActiveCircuit
        index = lines.First
        while index:
            branch = self._branch_by_key.get(lines.Name.lower(), {})
            records.append(self._element_record(lines.Name, str(branch.get("kind", "line")), lines.Bus1, lines.Bus2, branch, circuit.ActiveCktElement))
            index = lines.Next
        index = transformers.First
        while index:
            branch = self._branch_by_key.get(transformers.Name.lower(), {})
            buses = tuple(circuit.ActiveCktElement.BusNames)
            bus1, bus2 = (buses + ("", ""))[:2]
            kind = str(branch.get("kind", "transformer"))
            transformers.Wdg = 2
            tap_pu = float(transformers.Tap)
            tap = branch.get("tap", {})
            tap_position = (tap_pu - float(tap.get("min_pu", 1.0))) / ((float(tap.get("max_pu", 1.0)) - float(tap.get("min_pu", 1.0))) / max(int(tap.get("num_taps", 1)), 1)) - int(tap.get("num_taps", 1)) / 2 if "tap" in branch else float("nan")
            records.append(self._element_record(transformers.Name, kind, bus1, bus2, branch, circuit.ActiveCktElement, tap_pu, tap_position))
            index = transformers.Next
        return records

    @staticmethod
    def _element_record(name: str, kind: str, bus1: str, bus2: str, branch: Mapping[str, Any], element: Any, tap_pu: float = float("nan"), tap_position: float = float("nan")) -> dict[str, Any]:
        ncond = int(element.NumConductors)
        currents, powers = element.CurrentsMagAng, element.Powers
        count = min(ncond, len(currents) // 2, len(powers) // 2)
        phase_powers = [complex(float(powers[2 * item]), float(powers[2 * item + 1])) for item in range(count)]
        terminal_p = sum(value.real for value in phase_powers)
        terminal_q = sum(value.imag for value in phase_powers)
        max_current = max((float(currents[2 * item]) for item in range(count)), default=0.0)
        return {
            "branch": name,
            "branch_kind": kind,
            "bus1": bus1,
            "bus2": bus2,
            "max_current_a": max_current,
            "tap_pu": tap_pu,
            "tap_position": tap_position,
            "terminal1_p_kw": terminal_p,
            "terminal1_q_kvar": terminal_q,
            "terminal1_apparent_kva": sum(abs(value) for value in phase_powers),
        }

    @staticmethod
    def _system_summary(records: list[dict[str, Any]]) -> dict[str, float | bool]:
        circuit = dss.ActiveCircuit
        loss_w, loss_var = circuit.Losses
        total_p_kw, total_q_kvar = circuit.TotalPower
        magnitudes = [record["v_pu"] for record in records if record["v_pu"] > 0]
        return {"converged": bool(circuit.Solution.Converged), "min_v_pu": min(magnitudes), "max_v_pu": max(magnitudes), "loss_kw": float(loss_w) / 1000.0, "loss_kvar": float(loss_var) / 1000.0, "source_p_kw": float(total_p_kw), "source_q_kvar": float(total_q_kvar)}
