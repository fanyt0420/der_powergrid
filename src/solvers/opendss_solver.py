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
            dss(f"Storage.{name}.State={state}")
            dss(f"Storage.{name}.kW={abs(command.p_kw)}")
            dss(f"Storage.{name}.kvar={command.q_kvar}")
            if "soc_pct" in command.parameters:
                dss(f"Storage.{name}.%stored={float(command.parameters['soc_pct'])}")
        else:
            raise ValueError(f"No OpenDSS mapping for DER kind {kind!r}.")

    def solve(self, operating_point: OperatingPoint, state: PFState | None, context: SolverContext | None = None) -> PFResult:
        self._network()
        self._set_loads(operating_point.load_pq_kva)
        for name, command in operating_point.der_commands.items():
            self._set_der_command(name, command)
        solution = dss.ActiveCircuit.Solution
        solution.Solve()
        if not solution.Converged:
            raise RuntimeError(f"OpenDSS power flow did not converge at hour {operating_point.hour}.")
        voltage_records, voltages = self._bus_voltage_records()
        line_records = self._line_records()
        return PFResult(
            converged=True,
            bus_voltages=voltages,
            bus_voltage_records=tuple(voltage_records),
            branch_records=tuple(line_records),
            summary=self._system_summary(voltage_records),
            iterations=int(solution.Iterations),
            next_state=PFState(voltage_guess=voltages),
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
