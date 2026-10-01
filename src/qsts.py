"""Solver-independent QSTS orchestration over dense profile arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.power_flow import ConstraintResult, HostingCapacityMetrics, NetworkModel, OperatingPoint, PFResult, PFState, PowerFlowSolver, SolverContext
from src.profile_store import ProfileStore


@dataclass(frozen=True)
class SecurityEvaluation:
    """Solver-independent security and headroom results for one QSTS point."""

    constraints: ConstraintResult
    hosting_capacity: HostingCapacityMetrics
    branch_records: tuple[Mapping[str, Any], ...]

    def summary(self) -> Mapping[str, float | bool | str]:
        return {**self.constraints.summary(), **self.hosting_capacity.summary()}


class SecurityEvaluator:
    """Apply one common constraint definition to results from any PF solver."""

    def __init__(self, network: NetworkModel) -> None:
        self.network = network
        self.limits = {
            "voltage_min_pu": .95, "voltage_max_pu": 1.05,
            "voltage_unbalance_limit_pct": 2.0, "line_loading_limit_pct": 100.0,
            "transformer_loading_limit_pct": 100.0, "reverse_power_tolerance_kw": 1e-6,
            **network.constraints,
        }

    def evaluate(self, result: PFResult) -> SecurityEvaluation:
        records = tuple(self._enrich_branch(record) for record in result.branch_records)
        magnitudes = np.fromiter((float(record["v_pu"]) for record in result.bus_voltage_records if record["v_pu"] > 0), dtype=float)
        unbalance = self._voltage_unbalance_pct(result.bus_voltages)
        line = np.fromiter((float(record["loading_pct"]) for record in records if record["branch_kind"] in {"line", "switch"} and np.isfinite(record["loading_pct"])), dtype=float)
        transformer = np.fromiter((float(record["loading_pct"]) for record in records if record["branch_kind"] in {"transformer", "regulator"} and np.isfinite(record["loading_pct"])), dtype=float)
        max_line = float(np.max(line)) if len(line) else float("nan")
        max_transformer = float(np.max(transformer)) if len(transformer) else float("nan")
        source_p = float(result.summary["source_p_kw"])
        reverse_kw = max(source_p, 0.0)
        constraints = ConstraintResult(
            voltage_low_count=int(np.sum(magnitudes < float(self.limits["voltage_min_pu"]))),
            voltage_high_count=int(np.sum(magnitudes > float(self.limits["voltage_max_pu"]))),
            voltage_unbalance_count=int(np.sum(unbalance > float(self.limits["voltage_unbalance_limit_pct"]))),
            line_thermal_violation_count=int(np.sum(line > float(self.limits["line_loading_limit_pct"]))),
            transformer_thermal_violation_count=int(np.sum(transformer > float(self.limits["transformer_loading_limit_pct"]))),
            min_voltage_pu=float(np.min(magnitudes)), max_voltage_pu=float(np.max(magnitudes)),
            max_voltage_unbalance_pct=float(np.max(unbalance)) if len(unbalance) else 0.0,
            max_line_loading_pct=max_line, max_transformer_loading_pct=max_transformer,
            reverse_power_flow=reverse_kw > float(self.limits["reverse_power_tolerance_kw"]), reverse_power_kw=reverse_kw,
        )
        voltage_headroom = min(constraints.min_voltage_pu - float(self.limits["voltage_min_pu"]), float(self.limits["voltage_max_pu"]) - constraints.max_voltage_pu)
        line_headroom = float(self.limits["line_loading_limit_pct"]) - max_line if np.isfinite(max_line) else float("nan")
        transformer_headroom = float(self.limits["transformer_loading_limit_pct"]) - max_transformer if np.isfinite(max_transformer) else float("nan")
        normalized = {"voltage": voltage_headroom / .05, "line": line_headroom / 100.0 if np.isfinite(line_headroom) else float("inf"), "transformer": transformer_headroom / 100.0 if np.isfinite(transformer_headroom) else float("inf")}
        hosting = HostingCapacityMetrics(voltage_headroom, line_headroom, transformer_headroom, min(normalized, key=normalized.get))
        return SecurityEvaluation(constraints, hosting, records)

    def _enrich_branch(self, record: Mapping[str, Any]) -> dict[str, Any]:
        raw = dict(record)
        branch = self.network.branches[self.network.branch_index[str(raw["branch"]).lower()]]
        kind = str(raw["branch_kind"])
        normal_amps = float(branch.get("normal_amps", float("nan")))
        rating_kva = float(sum(float(winding.get("kva", 0.0)) for winding in branch.get("windings", ())[:1]))
        if kind in {"line", "switch"}:
            loading = float(raw["max_current_a"]) / normal_amps * 100.0 if np.isfinite(normal_amps) and normal_amps > 0 else float("nan")
        else:
            loading = float(raw["terminal1_apparent_kva"]) / rating_kva * 100.0 if rating_kva > 0 else float("nan")
        limit = float(self.limits["line_loading_limit_pct"] if kind in {"line", "switch"} else self.limits["transformer_loading_limit_pct"])
        return {**raw, "normal_amps": normal_amps, "rating_kva": rating_kva, "loading_pct": loading, "thermal_violation": bool(np.isfinite(loading) and loading > limit), "reverse_power_flow": bool(float(raw["terminal1_p_kw"]) < -float(self.limits["reverse_power_tolerance_kw"]))}

    @staticmethod
    def _voltage_unbalance_pct(voltages: Mapping[str, complex]) -> np.ndarray:
        by_bus: dict[str, dict[int, complex]] = {}
        for key, value in voltages.items():
            bus, node = key.rsplit(".", 1)
            by_bus.setdefault(bus.lower(), {})[int(node)] = value
        a, values = complex(-.5, np.sqrt(3) / 2), []
        for phases in by_bus.values():
            if {1, 2, 3}.issubset(phases):
                positive = (phases[1] + a * phases[2] + a * a * phases[3]) / 3
                negative = (phases[1] + a * a * phases[2] + a * phases[3]) / 3
                values.append(100.0 * abs(negative) / max(abs(positive), 1e-12))
        return np.asarray(values)


def _record_buffer(records: tuple[dict[str, Any], ...], count: int, timestamp_name: str) -> dict[str, np.ndarray]:
    """Allocate column arrays once from a solver record schema."""
    keys = tuple(records[0]) if records else ()
    buffer: dict[str, np.ndarray] = {timestamp_name: np.empty(count * len(records), dtype=object)}
    for key in keys:
        sample = records[0][key]
        buffer[key] = np.empty(count * len(records), dtype=object if isinstance(sample, str) else float)
    return buffer


def _write_records(buffer: dict[str, np.ndarray], records: tuple[dict[str, Any], ...], time_index: int, timestamp: str) -> None:
    width = len(records)
    if not width:
        return
    offset = time_index * width
    section = slice(offset, offset + width)
    buffer["timestamp"][section] = timestamp
    for key in records[0]:
        buffer[key][section] = [record[key] for record in records]


def _final_commands_or_raise(result_commands: Mapping[str, Any], expected_ids: tuple[str, ...], timestamp: pd.Timestamp) -> Mapping[str, Any]:
    """Enforce the solver contract for commands after coupled PF/control solving."""
    expected, actual = set(expected_ids), set(result_commands)
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            "PowerFlowSolver must return final_der_commands for exactly every DER "
            f"at {timestamp}; missing={missing}, unexpected={unexpected}."
        )
    return result_commands


def run_qsts(
    solver: PowerFlowSolver,
    profiles: ProfileStore,
    network: NetworkModel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run QSTS while retaining only the time loop in Python.

    ``ProfileStore`` owns all load/DER inputs as validated ``T x N`` arrays. Each
    time step creates the solver-facing mappings required by the backend, advances
    BESS SOC for every BESS in one NumPy operation, and writes into preallocated
    result columns.
    """
    timestamps = profiles.timestamps
    count = len(timestamps)
    dt_hours = network.step_seconds() / 3600.0
    pf_state: PFState | None = None
    security = SecurityEvaluator(network)

    final_p = np.empty((count, len(profiles.der_ids)))
    final_q = np.empty((count, len(profiles.der_ids)))
    bess_indices = np.flatnonzero(profiles.bess_mask)
    bess_soc_history = np.empty((count, len(bess_indices)))
    control_iterations = np.empty(count, dtype=int)
    pf_solve_count = np.empty(count, dtype=int)
    pf_iterations_total = np.empty(count, dtype=int)
    pf_iterations = np.empty(count, dtype=int)
    summary_columns: dict[str, np.ndarray] | None = None
    bus_buffer: dict[str, np.ndarray] | None = None
    branch_buffer: dict[str, np.ndarray] | None = None

    for time_index, timestamp in enumerate(timestamps):
        commands = profiles.commands_at(time_index, dt_hours)
        result = solver.solve(
            operating_point=OperatingPoint(
                time=timestamp.to_pydatetime(),
                load_pq_kva=profiles.load_mapping_at(time_index),
                der_commands=commands,
            ),
            state=pf_state,
            context=SolverContext(time_step_hours=dt_hours),
        )
        final_commands = _final_commands_or_raise(result.final_der_commands, profiles.der_ids, timestamp)
        evaluation = security.evaluate(result)
        combined_summary = {**result.summary, **evaluation.summary()}
        final_p[time_index] = np.fromiter((final_commands[name].p_kw for name in profiles.der_ids), dtype=float, count=len(profiles.der_ids))
        final_q[time_index] = np.fromiter((final_commands[name].q_kvar for name in profiles.der_ids), dtype=float, count=len(profiles.der_ids))
        profiles.advance_bess(final_commands, dt_hours)
        if len(bess_indices):
            bess_soc_history[time_index] = profiles.soc_pct[bess_indices]
        pf_state = result.next_state

        if summary_columns is None:
            summary_columns = {key: np.empty(count, dtype=bool if isinstance(value, bool) else object if isinstance(value, str) else float) for key, value in combined_summary.items()}
            bus_buffer = _record_buffer(result.bus_voltage_records, count, "timestamp")
            branch_buffer = _record_buffer(evaluation.branch_records, count, "timestamp")
        for key, values in summary_columns.items():
            values[time_index] = combined_summary[key]
        control_iterations[time_index] = result.control_iterations
        pf_solve_count[time_index] = result.pf_solve_count
        pf_iterations_total[time_index] = result.pf_iterations_total
        pf_iterations[time_index] = result.pf_iterations_last
        timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        _write_records(bus_buffer, result.bus_voltage_records, time_index, timestamp_text)
        _write_records(branch_buffer, evaluation.branch_records, time_index, timestamp_text)

    assert summary_columns is not None and bus_buffer is not None and branch_buffer is not None
    system: dict[str, Any] = {
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "dt_hours": np.full(count, dt_hours),
        "total_load_kw": profiles.load_p_kw.sum(axis=1),
        "total_load_kvar": profiles.load_q_kvar.sum(axis=1),
        "control_iterations": control_iterations,
        "pf_solve_count": pf_solve_count,
        "pf_iterations_total": pf_iterations_total,
        "pf_iterations": pf_iterations,
        **summary_columns,
    }
    for column, name in enumerate(profiles.der_ids):
        prefix = name.lower()
        system[f"{prefix}_p_kw"] = final_p[:, column]
        system[f"{prefix}_q_kvar"] = final_q[:, column]
    for position, column in enumerate(bess_indices):
        system[f"{profiles.der_ids[column].lower()}_soc_pct"] = bess_soc_history[:, position]
    return pd.DataFrame(bus_buffer), pd.DataFrame(system), pd.DataFrame(branch_buffer)
