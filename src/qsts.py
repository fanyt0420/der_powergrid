from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.opendss_model import OpenDSSModel


def run_qsts(
    model: OpenDSSModel,
    load_profile_file: str | Path,
    dt_hours: float = 1.0,
    max_control_iterations: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run QSTS with an individual P/Q profile for every non-DER load.

    ``load_profile_file`` must contain ``hour, load_name, p_kw, q_kvar``. Every
    simulated hour must provide exactly one row for each non-DER OpenDSS Load.
    """
    profile = pd.read_csv(load_profile_file)
    required = {"hour", "load_name", "p_kw", "q_kvar"}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"Profile is missing columns: {sorted(missing)}")

    if profile.duplicated(["hour", "load_name"]).any():
        raise ValueError("Load profile contains duplicate (hour, load_name) rows.")

    voltage_rows: list[dict] = []
    system_rows: list[dict] = []

    for hour, hour_profile in profile.groupby("hour", sort=True):
        hour = int(hour)
        load_values = {
            str(row.load_name): (float(row.p_kw), float(row.q_kvar))
            for row in hour_profile.itertuples(index=False)
        }

        model.set_load_values(load_values)
        model.apply_der_injections(hour)
        model.solve()

        control_iterations = 0
        while control_iterations < max_control_iterations and model.apply_der_controllers():
            model.solve()
            control_iterations += 1
        if control_iterations == max_control_iterations:
            raise RuntimeError(f"DER controls did not converge at hour {hour}.")

        model.advance_der_states(dt_hours)

        for rec in model.bus_voltage_records():
            voltage_rows.append({"hour": hour, **rec})

        summary = model.system_summary()
        system_rows.append(
            {
                "hour": hour,
                "total_load_kw": sum(p_kw for p_kw, _ in load_values.values()),
                "total_load_kvar": sum(q_kvar for _, q_kvar in load_values.values()),
                "control_iterations": control_iterations,
                **summary,
                **model.der_injection_summary(),
                **model.der_state_summary(),
            }
        )

    return pd.DataFrame(voltage_rows), pd.DataFrame(system_rows)
