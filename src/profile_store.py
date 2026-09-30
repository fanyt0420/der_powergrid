"""Dense, indexed time-series storage for QSTS load and DER inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.power_flow import DERCommand, NetworkModel


_POWER_COLUMNS = {"pv": ("pv_kw", "pv_kvar"), "wind": ("wind_kw", "wind_kvar"), "bess": ("bess_kw", "bess_kvar"), "ev": ("ev_kw", "ev_kvar")}
_CONTROLLERS = {"constant_pq", "volt_var", "soc_schedule"}


@dataclass
class ProfileStore:
    """Validated ``T x N`` arrays with immutable device/load ordering.

    DataFrames are only used while files are read. The QSTS loop accesses dense
    NumPy rows by integer time/device/load index and retains no per-device frame.
    """

    timestamps: pd.DatetimeIndex
    load_ids: tuple[str, ...]
    der_ids: tuple[str, ...]
    der_kinds: np.ndarray
    der_buses: tuple[str, ...]
    der_phases: tuple[tuple[int, ...], ...]
    load_p_kw: np.ndarray
    load_q_kvar: np.ndarray
    der_p_kw: np.ndarray
    der_q_kvar: np.ndarray
    controller_types: np.ndarray
    q_limit_kvar: np.ndarray
    v_ref_pu: np.ndarray
    droop_kvar_per_pu: np.ndarray
    bess_mask: np.ndarray
    bess_capacity_kwh: np.ndarray
    bess_p_rated_kw: np.ndarray
    bess_reserve_soc: np.ndarray
    bess_charge_efficiency: np.ndarray
    bess_discharge_efficiency: np.ndarray
    soc_pct: np.ndarray

    @classmethod
    def from_case(cls, network: NetworkModel, data_dir: str | Path) -> "ProfileStore":
        data_dir = Path(data_dir)
        timestamps = pd.DatetimeIndex(network.simulation_timestamps())
        load_indices = network.load_indices
        der_indices = network.der_indices
        load_ids = tuple(network.device_ids[index] for index in load_indices)
        der_ids = tuple(network.device_ids[index] for index in der_indices)
        t_count, load_count, der_count = len(timestamps), len(load_ids), len(der_ids)

        load_p, load_q = cls._load_arrays(data_dir / "load_profiles.csv", timestamps, load_ids)
        der_p, der_q = np.empty((t_count, der_count)), np.empty((t_count, der_count))
        controller = np.empty((t_count, der_count), dtype=object)
        q_limit = np.zeros((t_count, der_count))
        v_ref = np.ones((t_count, der_count))
        droop = np.zeros((t_count, der_count))
        kinds = np.empty(der_count, dtype=object)
        buses: list[str] = []
        phases: list[tuple[int, ...]] = []
        capacity = np.zeros(der_count)
        p_rated = np.zeros(der_count)
        reserve = np.zeros(der_count)
        charge_eff = np.ones(der_count)
        discharge_eff = np.ones(der_count)
        soc = np.full(der_count, np.nan)

        for column, device_index in enumerate(der_indices):
            device = network.devices[device_index]
            kind = str(device["kind"])
            if kind not in _POWER_COLUMNS:
                raise ValueError(f"Unsupported DER kind {kind!r} for device {device['id']!r}.")
            kinds[column] = kind
            buses.append(str(device["bus"]))
            phases.append(tuple(int(node) for node in device.get("phases", ())))
            profile_path = data_dir / str(device["profile_file"])
            values = cls._der_arrays(profile_path, timestamps, kind, str(device["id"]))
            der_p[:, column], der_q[:, column], controller[:, column], q_limit[:, column], v_ref[:, column], droop[:, column] = values
            if kind == "bess":
                capacity[column] = float(device["energy_kwh"])
                p_rated[column] = float(device.get("p_rated_kw", 0.0))
                reserve[column] = float(device.get("reserve_soc", 20.0))
                charge_eff[column] = float(device.get("charge_efficiency", 0.95))
                discharge_eff[column] = float(device.get("discharge_efficiency", 0.95))
                soc[column] = float(device.get("initial_soc", 50.0))

        return cls(
            timestamps, load_ids, der_ids, kinds, tuple(buses), tuple(phases), load_p, load_q,
            der_p, der_q, controller, q_limit, v_ref, droop, kinds == "bess", capacity,
            p_rated, reserve, charge_eff, discharge_eff, soc,
        )

    @staticmethod
    def _time_rows(frame: pd.DataFrame, timestamps: pd.DatetimeIndex, source: str) -> np.ndarray:
        actual = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
        if actual.duplicated().any():
            raise ValueError(f"{source} contains duplicate timestamps.")
        positions = timestamps.get_indexer(actual)
        if (positions < 0).any() or len(actual) != len(timestamps):
            expected = set(timestamps.strftime("%Y-%m-%d %H:%M:%S"))
            found = set(actual.strftime("%Y-%m-%d %H:%M:%S"))
            raise ValueError(f"{source} does not align with the simulation timeline: missing={sorted(expected - found)}, extra={sorted(found - expected)}.")
        return positions

    @classmethod
    def _load_arrays(cls, path: Path, timestamps: pd.DatetimeIndex, load_ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        frame = pd.read_csv(path)
        required = {"timestamp", "load_name", "p_kw", "q_kvar"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Load profile is missing columns: {sorted(missing)}")
        if frame.duplicated(["timestamp", "load_name"]).any():
            raise ValueError("Load profile contains duplicate (timestamp, load_name) rows.")
        id_index = {name.lower(): column for column, name in enumerate(load_ids)}
        names = frame["load_name"].astype(str).str.lower()
        columns = names.map(id_index)
        if columns.isna().any():
            raise ValueError(f"Load profile contains unknown loads: {sorted(frame.loc[columns.isna(), 'load_name'].astype(str).unique())}")
        rows = timestamps.get_indexer(pd.DatetimeIndex(pd.to_datetime(frame["timestamp"])))
        if (rows < 0).any() or len(frame) != len(timestamps) * len(load_ids):
            raise ValueError("Load profile does not contain exactly one row for every (timestamp, load_name).")
        p, q = np.empty((len(timestamps), len(load_ids))), np.empty((len(timestamps), len(load_ids)))
        p[rows, columns.to_numpy(dtype=int)] = frame["p_kw"].to_numpy(float)
        q[rows, columns.to_numpy(dtype=int)] = frame["q_kvar"].to_numpy(float)
        return p, q

    @classmethod
    def _der_arrays(cls, path: Path, timestamps: pd.DatetimeIndex, kind: str, device_id: str) -> tuple[np.ndarray, ...]:
        if not path.exists():
            raise FileNotFoundError(f"DER profile not found: {path}")
        frame = pd.read_csv(path)
        p_column, q_column = _POWER_COLUMNS[kind]
        required = {"timestamp", p_column, "controller_type"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"DER {device_id!r} profile is missing columns: {sorted(missing)}")
        rows = cls._time_rows(frame, timestamps, f"DER {device_id!r} profile")
        controller = frame["controller_type"].astype(str).to_numpy(object)
        unsupported = set(controller).difference(_CONTROLLERS)
        if unsupported:
            raise ValueError(f"DER {device_id!r} profile has unsupported controller types: {sorted(unsupported)}")
        p = np.empty(len(timestamps)); q = np.empty(len(timestamps)); types = np.empty(len(timestamps), dtype=object)
        limit = np.zeros(len(timestamps)); ref = np.ones(len(timestamps)); droop = np.zeros(len(timestamps))
        p[rows] = frame[p_column].to_numpy(float)
        q[rows] = frame[q_column].to_numpy(float) if q_column in frame else 0.0
        types[rows] = controller
        limit[rows] = frame.get("q_limit_kvar", 0.0)
        ref[rows] = frame.get("v_ref_pu", 1.0)
        droop[rows] = frame.get("droop_kvar_per_pu", 0.0)
        return p, q, types, limit, ref, droop

    def commands_at(self, time_index: int, dt_hours: float) -> dict[str, DERCommand]:
        p = self.der_p_kw[time_index].copy()
        bess = self.bess_mask
        if np.any(bess):
            p[bess] = np.clip(p[bess], -self.bess_p_rated_kw[bess], self.bess_p_rated_kw[bess])
            discharge_limit = (self.soc_pct[bess] - self.bess_reserve_soc[bess]) / 100.0 * self.bess_capacity_kwh[bess] * self.bess_discharge_efficiency[bess] / dt_hours
            charge_limit = (100.0 - self.soc_pct[bess]) / 100.0 * self.bess_capacity_kwh[bess] / self.bess_charge_efficiency[bess] / dt_hours
            requested = p[bess]
            p[bess] = np.where(requested > 0.0, np.minimum(requested, discharge_limit), np.where(requested < 0.0, -np.minimum(-requested, charge_limit), requested))
        q = self.der_q_kvar[time_index]
        types = self.controller_types[time_index]
        return {
            device_id: DERCommand(
                float(p[column]), float(q[column]), str(types[column]),
                {
                    "q_limit_kvar": float(self.q_limit_kvar[time_index, column]),
                    "v_ref_pu": float(self.v_ref_pu[time_index, column]),
                    "droop_kvar_per_pu": float(self.droop_kvar_per_pu[time_index, column]),
                    "terminal_nodes": self.der_phases[column],
                    **({"soc_pct": float(self.soc_pct[column])} if self.bess_mask[column] else {}),
                },
            )
            for column, device_id in enumerate(self.der_ids)
        }

    def load_mapping_at(self, time_index: int) -> dict[str, complex]:
        return dict(zip(self.load_ids, self.load_p_kw[time_index] + 1j * self.load_q_kvar[time_index], strict=True))

    def advance_bess(self, final_commands: Mapping[str, DERCommand], dt_hours: float) -> None:
        if not np.any(self.bess_mask):
            return
        final_p = np.fromiter((final_commands[name].p_kw for name in self.der_ids), dtype=float, count=len(self.der_ids))
        idx = np.flatnonzero(self.bess_mask)
        p = final_p[idx]
        delta = np.where(p > 0.0, -p * dt_hours / self.bess_discharge_efficiency[idx] / self.bess_capacity_kwh[idx] * 100.0, np.where(p < 0.0, -p * dt_hours * self.bess_charge_efficiency[idx] / self.bess_capacity_kwh[idx] * 100.0, 0.0))
        self.soc_pct[idx] = np.clip(self.soc_pct[idx] + delta, self.bess_reserve_soc[idx], 100.0)

    def state_summary(self) -> dict[str, float]:
        return {f"{self.der_ids[index].lower()}_soc_pct": float(self.soc_pct[index]) for index in np.flatnonzero(self.bess_mask)}
