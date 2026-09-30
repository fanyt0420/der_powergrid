"""Solver-independent QSTS orchestration and DER-control coupling."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.der_model import DERModel
from src.power_flow import NetworkModel, OperatingPoint, PFState, PowerFlowSolver, SolverContext


def _ts_str(ts: pd.Timestamp) -> str:
    """Normalize a timestamp to the string form stored in profile CSVs."""
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def run_qsts(
    solver: PowerFlowSolver,
    der_models: Sequence[DERModel],
    load_profile_file: str | Path,
    network: NetworkModel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run QSTS through a backend-neutral PowerFlowSolver.

    The authoritative timeline comes from ``network.simulation`` (start/end/step);
    the load and every DER profile must align with it exactly. For each timestamp QSTS
    builds an :class:`OperatingPoint` and calls the solver exactly once, then advances
    cross-time state using the solver's ``final_der_commands``.
    """
    timestamps = network.simulation_timestamps()
    step_seconds = network.step_seconds()
    dt_hours = step_seconds / 3600.0

    profile = pd.read_csv(load_profile_file)
    required = {"timestamp", "load_name", "p_kw", "q_kvar"}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"Profile is missing columns: {sorted(missing)}")

    profile["timestamp"] = pd.to_datetime(profile["timestamp"])
    if profile.duplicated(["timestamp", "load_name"]).any():
        raise ValueError("Load profile contains duplicate (timestamp, load_name) rows.")

    # 负荷 profile 与时间轴对齐校验
    expected = {_ts_str(t) for t in timestamps}
    actual = set(profile["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    missing_ts = sorted(expected - actual)
    extra_ts = sorted(actual - expected)
    if missing_ts or extra_ts:
        raise ValueError(
            f"Load profile does not align with the simulation timeline: "
            f"missing={missing_ts}, extra={extra_ts}."
        )

    # 每个时间点上，负荷名必须覆盖全部基础负荷且恰好一个
    expected_loads = network.base_load_names()
    for ts in timestamps:
        names = set(profile[profile["timestamp"] == ts]["load_name"].astype(str))
        if names != expected_loads:
            raise ValueError(f"Load profile at {_ts_str(ts)} has loads {sorted(names)}, expected {sorted(expected_loads)}.")

    # 校验每台 DER 的 profile 与时间轴对齐
    for model in der_models:
        model.validate_timeline(timestamps)

    voltage_rows: list[dict] = []
    system_rows: list[dict] = []
    branch_rows: list[dict] = []
    pf_state: PFState | None = None

    for ts in timestamps:
        point = profile[profile["timestamp"] == ts]
        loads = {str(row.load_name): complex(float(row.p_kw), float(row.q_kvar)) for row in point.itertuples(index=False)}

        injections = {model.name: model.get_injection(ts, dt_hours) for model in der_models}
        commands = {model.name: model.to_command(injections[model.name]) for model in der_models}
        time = ts.to_pydatetime()
        operating_point = OperatingPoint(time=time, load_pq_kva=loads, der_commands=commands)

        result = solver.solve(operating_point, pf_state, SolverContext(time_step_hours=dt_hours))

        final_commands = result.final_der_commands or commands
        # 使用最终收敛命令推进跨时刻状态
        for model in der_models:
            final_command = final_commands[model.name]
            model.advance_state(
                {"p_kw": final_command.p_kw, "q_kvar": final_command.q_kvar},
                dt_hours,
            )
        pf_state = result.next_state

        ts_text = _ts_str(ts)
        voltage_rows.extend({"timestamp": ts_text, **record} for record in result.bus_voltage_records)
        branch_rows.extend({"timestamp": ts_text, **record} for record in result.branch_records)
        injection_summary = {
            f"{name.lower()}_{quantity}": getattr(command, quantity)
            for name, command in final_commands.items()
            for quantity in ("p_kw", "q_kvar")
        }
        state_summary = {key: value for model in der_models for key, value in model.state_summary().items()}
        system_rows.append({
            "timestamp": ts_text,
            "dt_hours": dt_hours,
            "total_load_kw": sum(value.real for value in loads.values()),
            "total_load_kvar": sum(value.imag for value in loads.values()),
            "control_iterations": result.control_iterations,
            "pf_solve_count": result.pf_solve_count,
            "pf_iterations_total": result.pf_iterations_total,
            "pf_iterations": result.pf_iterations_last,
            **result.summary,
            **injection_summary,
            **state_summary,
        })

    return pd.DataFrame(voltage_rows), pd.DataFrame(system_rows), pd.DataFrame(branch_rows)