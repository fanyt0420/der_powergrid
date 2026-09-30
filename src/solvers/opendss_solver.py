"""OpenDSS implementation of the solver-neutral PowerFlowSolver interface."""

from __future__ import annotations

from cmath import rect
from math import radians
from pathlib import Path
from typing import Any, Mapping

from dss import dss
import numpy as np

from src.power_flow import ConstraintResult, DERCommand, HostingCapacityMetrics, NetworkModel, OperatingPoint, PFResult, PFState, SolverContext


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
        constraints, hosting = self._evaluate_constraints(voltage_records, voltages, branch_records, summary)
        return PFResult(
            converged=True,
            bus_voltages=voltages,
            bus_voltage_records=tuple(voltage_records),
            branch_records=tuple(branch_records),
            summary={**summary, **constraints.summary(), **hosting.summary()},
            iterations=pf_iterations_last,
            next_state=PFState(voltage_guess=voltages),
            final_der_commands=commands,
            control_iterations=control_iterations,
            pf_solve_count=pf_solve_count,
            pf_iterations_total=pf_iterations_total,
            pf_iterations_last=pf_iterations_last,
            metadata={"backend": "opendss", "context": context},
            constraints=constraints,
            hosting_capacity=hosting,
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
        normal_amps = float(branch.get("normal_amps", float("nan")))
        rating_kva = float(sum(float(winding.get("kva", 0.0)) for winding in branch.get("windings", ())[:1]))
        if kind == "line" or kind == "switch":
            loading = max_current / normal_amps * 100.0 if np.isfinite(normal_amps) and normal_amps > 0 else float("nan")
        else:
            apparent_kva = sum(abs(value) for value in phase_powers)
            loading = apparent_kva / rating_kva * 100.0 if rating_kva > 0 else float("nan")
        return {
            "branch": name,
            "branch_kind": kind,
            "bus1": bus1,
            "bus2": bus2,
            "max_current_a": max_current,
            "normal_amps": normal_amps,
            "rating_kva": rating_kva,
            "loading_pct": loading,
            "thermal_violation": bool(np.isfinite(loading) and loading > 100.0),
            "tap_pu": tap_pu,
            "tap_position": tap_position,
            "terminal1_p_kw": terminal_p,
            "terminal1_q_kvar": terminal_q,
            "reverse_power_flow": bool(terminal_p < -1e-6),
        }

    def _evaluate_constraints(
        self,
        voltage_records: list[dict[str, Any]],
        voltages: Mapping[str, complex],
        branch_records: list[dict[str, Any]],
        system: Mapping[str, float | bool],
    ) -> tuple[ConstraintResult, HostingCapacityMetrics]:
        limits = {"voltage_min_pu": .95, "voltage_max_pu": 1.05, "voltage_unbalance_limit_pct": 2.0, "line_loading_limit_pct": 100.0, "transformer_loading_limit_pct": 100.0, "reverse_power_tolerance_kw": 1e-6, **self._network().constraints}
        magnitudes = np.fromiter((float(record["v_pu"]) for record in voltage_records if record["v_pu"] > 0), dtype=float)
        unbalance = self._voltage_unbalance_pct(voltages)
        line_loading = np.fromiter((float(record["loading_pct"]) for record in branch_records if record["branch_kind"] in {"line", "switch"} and np.isfinite(record["loading_pct"])), dtype=float)
        transformer_loading = np.fromiter((float(record["loading_pct"]) for record in branch_records if record["branch_kind"] in {"transformer", "regulator"} and np.isfinite(record["loading_pct"])), dtype=float)
        max_line = float(np.max(line_loading)) if len(line_loading) else float("nan")
        max_transformer = float(np.max(transformer_loading)) if len(transformer_loading) else float("nan")
        max_unbalance = float(np.max(unbalance)) if len(unbalance) else 0.0
        source_p = float(system["source_p_kw"])
        reverse_kw = max(source_p, 0.0)
        constraints = ConstraintResult(
            voltage_low_count=int(np.sum(magnitudes < float(limits["voltage_min_pu"]))),
            voltage_high_count=int(np.sum(magnitudes > float(limits["voltage_max_pu"]))),
            voltage_unbalance_count=int(np.sum(unbalance > float(limits["voltage_unbalance_limit_pct"]))),
            line_thermal_violation_count=int(np.sum(line_loading > float(limits["line_loading_limit_pct"]))),
            transformer_thermal_violation_count=int(np.sum(transformer_loading > float(limits["transformer_loading_limit_pct"]))),
            min_voltage_pu=float(np.min(magnitudes)), max_voltage_pu=float(np.max(magnitudes)),
            max_voltage_unbalance_pct=max_unbalance, max_line_loading_pct=max_line,
            max_transformer_loading_pct=max_transformer,
            reverse_power_flow=bool(reverse_kw > float(limits["reverse_power_tolerance_kw"])), reverse_power_kw=reverse_kw,
        )
        voltage_headroom = min(constraints.min_voltage_pu - float(limits["voltage_min_pu"]), float(limits["voltage_max_pu"]) - constraints.max_voltage_pu)
        line_headroom = float(limits["line_loading_limit_pct"]) - max_line if np.isfinite(max_line) else float("nan")
        transformer_headroom = float(limits["transformer_loading_limit_pct"]) - max_transformer if np.isfinite(max_transformer) else float("nan")
        normalized = {"voltage": voltage_headroom / .05, "line": line_headroom / 100.0 if np.isfinite(line_headroom) else float("inf"), "transformer": transformer_headroom / 100.0 if np.isfinite(transformer_headroom) else float("inf")}
        hosting = HostingCapacityMetrics(voltage_headroom_pu=voltage_headroom, line_loading_headroom_pct=line_headroom, transformer_loading_headroom_pct=transformer_headroom, binding_constraint=min(normalized, key=normalized.get))
        return constraints, hosting

    @staticmethod
    def _voltage_unbalance_pct(voltages: Mapping[str, complex]) -> np.ndarray:
        by_bus: dict[str, dict[int, complex]] = {}
        for key, value in voltages.items():
            bus, node = key.rsplit(".", 1)
            by_bus.setdefault(bus.lower(), {})[int(node)] = value
        a = complex(-.5, np.sqrt(3) / 2)
        values = []
        for phase_values in by_bus.values():
            if {1, 2, 3}.issubset(phase_values):
                positive = (phase_values[1] + a * phase_values[2] + a * a * phase_values[3]) / 3
                negative = (phase_values[1] + a * a * phase_values[2] + a * phase_values[3]) / 3
                values.append(100.0 * abs(negative) / max(abs(positive), 1e-12))
        return np.asarray(values)

    @staticmethod
    def _system_summary(records: list[dict[str, Any]]) -> dict[str, float | bool]:
        circuit = dss.ActiveCircuit
        loss_w, loss_var = circuit.Losses
        total_p_kw, total_q_kvar = circuit.TotalPower
        magnitudes = [record["v_pu"] for record in records if record["v_pu"] > 0]
        return {"converged": bool(circuit.Solution.Converged), "min_v_pu": min(magnitudes), "max_v_pu": max(magnitudes), "loss_kw": float(loss_w) / 1000.0, "loss_kvar": float(loss_var) / 1000.0, "source_p_kw": float(total_p_kw), "source_q_kvar": float(total_q_kvar)}
