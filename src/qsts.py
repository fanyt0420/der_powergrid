"""Solver-independent QSTS orchestration over dense profile arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.power_flow import NetworkModel, OperatingPoint, PFState, PowerFlowSolver, SolverContext
from src.profile_store import ProfileStore


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
        final_commands = result.final_der_commands or commands
        final_p[time_index] = np.fromiter((final_commands[name].p_kw for name in profiles.der_ids), dtype=float, count=len(profiles.der_ids))
        final_q[time_index] = np.fromiter((final_commands[name].q_kvar for name in profiles.der_ids), dtype=float, count=len(profiles.der_ids))
        profiles.advance_bess(final_commands, dt_hours)
        if len(bess_indices):
            bess_soc_history[time_index] = profiles.soc_pct[bess_indices]
        pf_state = result.next_state

        if summary_columns is None:
            summary_columns = {key: np.empty(count, dtype=bool if isinstance(value, bool) else object if isinstance(value, str) else float) for key, value in result.summary.items()}
            bus_buffer = _record_buffer(result.bus_voltage_records, count, "timestamp")
            branch_buffer = _record_buffer(result.branch_records, count, "timestamp")
        for key, values in summary_columns.items():
            values[time_index] = result.summary[key]
        control_iterations[time_index] = result.control_iterations
        pf_solve_count[time_index] = result.pf_solve_count
        pf_iterations_total[time_index] = result.pf_iterations_total
        pf_iterations[time_index] = result.pf_iterations_last
        timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        _write_records(bus_buffer, result.bus_voltage_records, time_index, timestamp_text)
        _write_records(branch_buffer, result.branch_records, time_index, timestamp_text)

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
