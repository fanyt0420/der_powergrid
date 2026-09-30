"""Solver-independent QSTS orchestration and DER-control coupling."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.der_model import DERModel
from src.power_flow import OperatingPoint, PFState, PowerFlowSolver, SolverContext


def _ts_str(ts: pd.Timestamp) -> str:
    """Normalize a timestamp to the string form stored in profile CSVs."""
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def run_qsts(
    solver: PowerFlowSolver,
    der_models: Sequence[DERModel],
    load_profile_file: str | Path,
    max_control_iterations: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run QSTS through a backend-neutral PowerFlowSolver.

    Time is taken from the ``timestamp`` column of the load profile rather than a
    fixed hour index. The time step ``dt_hours`` is inferred per point from the
    distance to the next timestamp, so neither the solver nor the DER models assume
    any fixed step size.
    """
    profile = pd.read_csv(load_profile_file)
    required = {"timestamp", "load_name", "p_kw", "q_kvar"}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"Profile is missing columns: {sorted(missing)}")

    profile["timestamp"] = pd.to_datetime(profile["timestamp"])
    if profile.duplicated(["timestamp", "load_name"]).any():
        raise ValueError("Load profile contains duplicate (timestamp, load_name) rows.")

    timestamps = sorted(profile["timestamp"].unique())
    if not timestamps:
        raise ValueError("Load profile contains no timestamps.")

    voltage_rows: list[dict] = []
    system_rows: list[dict] = []
    pf_state: PFState | None = None

    for index, ts in enumerate(timestamps):
        if index + 1 < len(timestamps):
            dt_hours = (timestamps[index + 1] - ts).total_seconds() / 3600.0
        elif len(timestamps) > 1:
            dt_hours = (timestamps[index] - timestamps[index - 1]).total_seconds() / 3600.0
        else:
            dt_hours = 1.0
        if dt_hours <= 0:
            raise ValueError(f"Non-positive time step between timestamps at order index {index}.")

        point = profile[profile["timestamp"] == ts]
        loads = {str(row.load_name): complex(float(row.p_kw), float(row.q_kvar)) for row in point.itertuples(index=False)}

        injections = {model.name: model.get_injection(ts, dt_hours) for model in der_models}
        commands = {model.name: model.to_command(injections[model.name]) for model in der_models}
        time = ts.to_pydatetime()
        operating_point = OperatingPoint(time=time, load_pq_kva=loads, der_commands=commands)

        result = solver.solve(operating_point, pf_state, SolverContext(time_step_hours=dt_hours))
        control_iterations = 0
        while control_iterations < max_control_iterations:
            updated_injections = {
                model.name: model.control_step(result, injections[model.name])
                for model in der_models
            }
            changed = any(
                abs(float(updated_injections[name]["p_kw"]) - commands[name].p_kw) > 1e-6
                or abs(float(updated_injections[name]["q_kvar"]) - commands[name].q_kvar) > 1e-6
                for name in commands
            )
            if not changed:
                break
            injections = updated_injections
            commands = {model.name: model.to_command(injections[model.name]) for model in der_models}
            operating_point = OperatingPoint(time=time, load_pq_kva=loads, der_commands=commands)
            control_iterations += 1
            result = solver.solve(
                operating_point,
                result.next_state,
                SolverContext(time_step_hours=dt_hours, control_iteration=control_iterations, previous_result=result),
            )
        else:
            raise RuntimeError(f"DER controls did not converge at timestamp {_ts_str(ts)}.")

        for model in der_models:
            model.advance_state(injections[model.name], dt_hours)
        pf_state = result.next_state

        ts_text = _ts_str(ts)
        voltage_rows.extend({"timestamp": ts_text, **record} for record in result.bus_voltage_records)
        injection_summary = {
            f"{name.lower()}_{quantity}": getattr(command, quantity)
            for name, command in commands.items()
            for quantity in ("p_kw", "q_kvar")
        }
        state_summary = {key: value for model in der_models for key, value in model.state_summary().items()}
        system_rows.append({
            "timestamp": ts_text,
            "dt_hours": dt_hours,
            "total_load_kw": sum(value.real for value in loads.values()),
            "total_load_kvar": sum(value.imag for value in loads.values()),
            "control_iterations": control_iterations,
            "pf_iterations": result.iterations,
            **result.summary,
            **injection_summary,
            **state_summary,
        })

    return pd.DataFrame(voltage_rows), pd.DataFrame(system_rows)