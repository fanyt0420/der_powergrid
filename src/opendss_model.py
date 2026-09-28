from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from dss import dss

from src.der_model import DERModel


class OpenDSSModel:
    """Small wrapper around DSS-Python with unified DER interface support."""

    def __init__(self, master_file: str | Path) -> None:
        self.master_file = Path(master_file).resolve()
        self._base_loads: Dict[str, tuple[float, float]] = {}
        self._der_models: List[DERModel] = []
        self._current_injections: Dict[str, Dict[str, Any]] = {}

    def compile(self) -> None:
        if not self.master_file.exists():
            raise FileNotFoundError(
                f"OpenDSS master file not found: {self.master_file}"
            )

        dss("Clear")
        dss(f'Redirect "{self.master_file}"')

        self._cache_base_loads()

    def _cache_base_loads(self) -> None:
        self._base_loads.clear()

        loads = dss.ActiveCircuit.Loads

        i = loads.First
        while i:
            name = loads.Name

            self._base_loads[name] = (
                float(loads.kW),
                float(loads.kvar),
            )

            i = loads.Next

    def solve(self) -> None:
        solution = dss.ActiveCircuit.Solution

        solution.Solve()

        if not solution.Converged:
            raise RuntimeError("OpenDSS power flow did not converge.")

    def add_der_model(self, der_model: DERModel) -> None:
        """Add a DER model to the simulation.

        Args:
            der_model: Instance of DERModel (PVModel, WindModel, BESSModel, etc.)
        """
        self._der_models.append(der_model)

    def base_load_names(self) -> list[str]:
        """Return loads driven by the load-profile input, excluding DER loads."""
        der_loads = {
            der.name.lower()
            for der in self._der_models
            if der.element_type.lower() == "load"
        }
        return [name for name in self._base_loads if name.lower() not in der_loads]

    def set_load_values(self, load_values: Dict[str, tuple[float, float]]) -> None:
        """Set P/Q for individually profiled, non-DER load elements."""
        expected = {name.lower(): name for name in self.base_load_names()}
        provided = {name.lower(): values for name, values in load_values.items()}
        unknown = sorted(set(provided).difference(expected))
        missing = sorted(set(expected).difference(provided))
        if unknown or missing:
            details = []
            if unknown:
                details.append(f"unknown loads: {unknown}")
            if missing:
                details.append(f"missing loads: {missing}")
            raise ValueError("Invalid individual load profile; " + "; ".join(details))

        loads = dss.ActiveCircuit.Loads
        for normalized_name, canonical_name in expected.items():
            p_kw, q_kvar = provided[normalized_name]
            loads.Name = canonical_name
            loads.kW = float(p_kw)
            loads.kvar = float(q_kvar)

    def apply_der_injections(self, time_step: int) -> None:
        """Apply all DER injections for a specific time step.

        This unified interface allows multiple DER types to be applied in a consistent way.

        Args:
            time_step: Current simulation time step (e.g., hour)
        """
        circuit = dss.ActiveCircuit

        self._current_injections.clear()
        for der_model in self._der_models:
            injection = der_model.get_injection(time_step)
            der_model.apply_to_opendss(circuit, injection)
            self._current_injections[der_model.name] = injection

    def apply_der_controllers(self, tolerance: float = 1e-6) -> bool:
        """Update voltage-dependent DER controls after a solved network state.

        Returns ``True`` when any P/Q command changed and another OpenDSS solve is
        required. This supplies a fixed-point coupling between DER control equations
        and the OpenDSS network power-flow solution.
        """
        circuit = dss.ActiveCircuit
        changed = False
        for der_model in self._der_models:
            previous = self._current_injections[der_model.name]
            updated = der_model.control_step(circuit, previous)
            if (
                abs(float(updated["p_kw"]) - float(previous["p_kw"])) > tolerance
                or abs(float(updated["q_kvar"]) - float(previous["q_kvar"])) > tolerance
            ):
                der_model.apply_to_opendss(circuit, updated)
                self._current_injections[der_model.name] = updated
                changed = True
        return changed

    def advance_der_states(self, dt_hours: float) -> None:
        """Persist each DER's post-solve state for the next QSTS time step."""
        for der_model in self._der_models:
            der_model.advance_state(self._current_injections[der_model.name], dt_hours)

    def der_state_summary(self) -> Dict[str, float]:
        """Collect persistent DER state values for QSTS output."""
        summary: Dict[str, float] = {}
        for der_model in self._der_models:
            summary.update(der_model.state_summary())
        return summary

    def der_injection_summary(self) -> Dict[str, float]:
        """Collect final, controller-adjusted DER P/Q commands for result output."""
        return {
            f"{name.lower()}_{quantity}": float(injection[quantity])
            for name, injection in self._current_injections.items()
            for quantity in ("p_kw", "q_kvar")
        }

    def bus_voltage_records(self) -> List[dict]:
        records: List[dict] = []

        circuit = dss.ActiveCircuit

        for bus_name in circuit.AllBusNames:
            circuit.SetActiveBus(bus_name)

            bus = circuit.ActiveBus

            nodes = bus.Nodes
            values = bus.puVmagAngle

            for idx, node in enumerate(nodes):
                records.append(
                    {
                        "bus": bus_name,
                        "node": int(node),
                        "v_pu": float(values[2 * idx]),
                        "angle_deg": float(values[2 * idx + 1]),
                    }
                )

        return records

    def line_records(self) -> List[dict]:
        records: List[dict] = []

        circuit = dss.ActiveCircuit
        lines = circuit.Lines

        i = lines.First

        while i:
            name = lines.Name
            bus1 = lines.Bus1
            bus2 = lines.Bus2

            element = circuit.ActiveCktElement

            ncond = element.NumConductors
            current_mag_angle = element.CurrentsMagAng
            powers = element.Powers

            terminal1_current_mags = [
                float(current_mag_angle[2 * k])
                for k in range(
                    min(
                        ncond,
                        len(current_mag_angle) // 2,
                    )
                )
            ]

            terminal1_p = sum(
                float(powers[2 * k])
                for k in range(
                    min(
                        ncond,
                        len(powers) // 2,
                    )
                )
            )

            terminal1_q = sum(
                float(powers[2 * k + 1])
                for k in range(
                    min(
                        ncond,
                        len(powers) // 2,
                    )
                )
            )

            records.append(
                {
                    "line": name,
                    "bus1": bus1,
                    "bus2": bus2,
                    "max_current_a": max(
                        terminal1_current_mags,
                        default=0.0,
                    ),
                    "terminal1_p_kw": terminal1_p,
                    "terminal1_q_kvar": terminal1_q,
                }
            )

            i = lines.Next

        return records

    def system_summary(self) -> dict:
        circuit = dss.ActiveCircuit
        solution = circuit.Solution

        voltages = [
            r["v_pu"]
            for r in self.bus_voltage_records()
            if r["v_pu"] > 0
        ]

        loss_w, loss_var = circuit.Losses
        total_p_kw, total_q_kvar = circuit.TotalPower

        return {
            "converged": bool(solution.Converged),
            "min_v_pu": min(voltages),
            "max_v_pu": max(voltages),
            "loss_kw": float(loss_w) / 1000.0,
            "loss_kvar": float(loss_var) / 1000.0,
            "source_p_kw": float(total_p_kw),
            "source_q_kvar": float(total_q_kvar),
        }
    
