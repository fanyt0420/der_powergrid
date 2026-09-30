"""OpenDSS implementation of the solver-neutral PowerFlowSolver interface."""

from __future__ import annotations

from cmath import rect
from math import radians
from pathlib import Path
from typing import Any, Mapping

from dss import dss

from src.power_flow import DERCommand, NetworkModel, OperatingPoint, PFResult, PFState, SolverContext


class OpenDSSSolver:
    """Adapter that applies an OperatingPoint to OpenDSS and solves it."""

    def __init__(self, master_file: str | Path) -> None:
        self.master_file = Path(master_file).resolve()
        self.network: NetworkModel | None = None

    def build(self, network: NetworkModel) -> None:
        if not self.master_file.exists():
            raise FileNotFoundError(f"OpenDSS master file not found: {self.master_file}")
        dss("Clear")
        dss(f'Redirect "{self.master_file}"')
        self.network = network

    def _network(self) -> NetworkModel:
        if self.network is None:
            raise RuntimeError("Call solver.build(network) before solver.solve(...).")
        return self.network

    def _set_loads(self, load_pq_kva: Mapping[str, complex]) -> None:
        expected = {name.lower(): name for name in self._network().base_load_names()}
        provided = {name.lower(): value for name, value in load_pq_kva.items()}
        unknown, missing = sorted(set(provided) - set(expected)), sorted(set(expected) - set(provided))
        if unknown or missing:
            raise ValueError(f"Invalid load operating point; unknown={unknown}, missing={missing}")
        loads = dss.ActiveCircuit.Loads
        for key, canonical_name in expected.items():
            value = complex(provided[key])
            loads.Name, loads.kW, loads.kvar = canonical_name, value.real, value.imag

    def _set_der_command(self, name: str, command: DERCommand) -> None:
        kind = str(self._network().device(name)["kind"])
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
            new_commands = dict(commands)
            for name, command in commands.items():
                if command.controller_type != "volt_var":
                    continue
                # 平均相电压作为控制输入
                prefix = f"{self._network().device(name)['bus'].lower()}."
                terminal_nodes = {int(node) for node in command.parameters.get("terminal_nodes", ())}
                values = [abs(value) for key, value in voltages.items() if key.lower().startswith(prefix) and (not terminal_nodes or int(key.rsplit(".", 1)[1]) in terminal_nodes)]
                if not values:
                    raise ValueError(
                        f"Volt-VAR DER {name!r} found no bus voltage for bus "
                        f"{self._network().device(name)['bus']!r} at timestamp {operating_point.time}."
                    )
                voltage_pu = sum(values) / len(values)
                q_kvar = command.adjusted_q(voltage_pu)
                # 可选松弛，relaxation=1.0 时等价于直接更新
                q_kvar = command.q_kvar + relaxation * (q_kvar - command.q_kvar)
                new_commands[name] = command.with_q(q_kvar)
            changed = any(
                abs(new_commands[name].q_kvar - commands[name].q_kvar) > tolerance
                for name in new_commands
                if commands[name].controller_type == "volt_var"
            )
            if not changed:
                break
            control_iterations += 1
            if control_iterations > max_control_iterations:
                raise RuntimeError(f"DER controls did not converge at timestamp {operating_point.time}.")
            # 关键：只要任意一个 Volt-VAR DER 未收敛，所有 Volt-VAR DER 同步更新
            commands = new_commands
            solve_once()

        voltage_records, voltages = self._bus_voltage_records()
        line_records = self._line_records()
        return PFResult(
            converged=True,
            bus_voltages=voltages,
            bus_voltage_records=tuple(voltage_records),
            branch_records=tuple(line_records),
            summary=self._system_summary(voltage_records),
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

    @staticmethod
    def _line_records() -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        lines, circuit = dss.ActiveCircuit.Lines, dss.ActiveCircuit
        index = lines.First
        while index:
            element = circuit.ActiveCktElement
            ncond = element.NumConductors
            currents, powers = element.CurrentsMagAng, element.Powers
            count = min(ncond, len(currents) // 2, len(powers) // 2)
            records.append({"line": lines.Name, "bus1": lines.Bus1, "bus2": lines.Bus2, "max_current_a": max((float(currents[2 * k]) for k in range(count)), default=0.0), "terminal1_p_kw": sum(float(powers[2 * k]) for k in range(count)), "terminal1_q_kvar": sum(float(powers[2 * k + 1]) for k in range(count))})
            index = lines.Next
        return records

    @staticmethod
    def _system_summary(records: list[dict[str, Any]]) -> dict[str, float | bool]:
        circuit = dss.ActiveCircuit
        loss_w, loss_var = circuit.Losses
        total_p_kw, total_q_kvar = circuit.TotalPower
        magnitudes = [record["v_pu"] for record in records if record["v_pu"] > 0]
        return {"converged": bool(circuit.Solution.Converged), "min_v_pu": min(magnitudes), "max_v_pu": max(magnitudes), "loss_kw": float(loss_w) / 1000.0, "loss_kvar": float(loss_var) / 1000.0, "source_p_kw": float(total_p_kw), "source_q_kvar": float(total_q_kvar)}
