from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd

from src.power_flow import DERCommand


class DERModel(ABC):
    """Abstract base class for Distributed Energy Resource models.

    This interface provides a unified way to handle different DER types (PV, Wind, BESS, EV, etc.)
    in QSTS simulations. Future spatio-temporal DER models can simply extend this class
    without modifying the QSTS core logic.

    The key design principles:
    1. Unified data interface via get_injection() method
    2. Time-series data loaded from standardized CSV format
    3. OpenDSS element management abstracted away from QSTS
    4. Easy to extend for new DER types
    """

    SUPPORTED_CONTROLLERS = {"constant_pq", "volt_var", "soc_schedule"}

    def __init__(
        self,
        name: str,
        profile_file: Optional[str | Path] = None,
        bus_name: Optional[str] = None,
        phases: Optional[list[int]] = None,
    ):
        """Initialize DER model.

        Args:
            name: Name of the DER device (e.g., "PV1", "Wind1")
            profile_file: Optional path to time-series profile CSV
        """
        self.name = name
        self.profile_file = Path(profile_file) if profile_file else None
        self.bus_name = bus_name
        self.phases = tuple(phases or ())
        self._profile_data: Optional[pd.DataFrame] = None

    def load_profile(self) -> None:
        """Load time-series profile data from CSV file."""
        if self.profile_file and self.profile_file.exists():
            self._profile_data = pd.read_csv(self.profile_file)
            self._validate_profile()

    def validate_timeline(self, timestamps: list[pd.Timestamp]) -> None:
        """Validate profile timestamps against the authoritative simulation timeline.

        Every timestamp of the timeline must have exactly one profile row and no other
        timestamp may be present. Missing or unexpected entries raise instead of
        silently defaulting to zero.
        """
        if self._profile_data is None:
            raise ValueError(f"DER {self.name!r} has no profile loaded.")
        data_ts = pd.to_datetime(self._profile_data["timestamp"])
        if data_ts.duplicated().any():
            raise ValueError(f"DER {self.name!r} profile has duplicate timestamps.")
        expected = {pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in timestamps}
        actual = set(data_ts.dt.strftime("%Y-%m-%d %H:%M:%S"))
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                f"DER {self.name!r} profile does not align with the simulation timeline: "
                f"missing={missing}, extra={extra}."
            )

    @abstractmethod
    def _validate_profile(self) -> None:
        """Validate that the loaded profile contains required columns."""
        pass

    @abstractmethod
    def get_injection(self, time_step: datetime, dt_hours: float) -> Dict[str, Any]:
        """Get DER injection values for a specific timestamp.

        Args:
            time_step: Current simulation timestamp.
            dt_hours: Duration of this time step in hours.

        Returns:
            Dictionary with injection parameters, typically:
            - 'p_kw': Active power in kW (positive = generation, negative = consumption)
            - 'q_kvar': Reactive power in kvar
            - Additional parameters as needed (SOC for BESS, etc.)
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this DER model."""
        return {
            "name": self.name,
            "has_profile": self._profile_data is not None,
            "bus_name": self.bus_name,
        }

    def _validate_controller_profile(self) -> None:
        """Validate the controller type included in every DER profile."""
        if self._profile_data is None:
            return
        required = {"timestamp", "controller_type"}
        missing = required.difference(self._profile_data.columns)
        if missing:
            raise ValueError(f"DER profile missing columns: {sorted(missing)}")
        unsupported = set(self._profile_data["controller_type"].dropna().unique()).difference(
            self.SUPPORTED_CONTROLLERS
        )
        if unsupported:
            raise ValueError(f"Unsupported controller types: {sorted(unsupported)}")

    def _row_at(self, time_step: datetime) -> pd.Series:
        """Return the profile row matching ``time_step``.

        Raises if the profile is not loaded or the timestamp is missing; profiles are
        validated upfront against the timeline so a missing row is an error, not a
        silent zero.
        """
        if self._profile_data is None:
            raise ValueError(f"DER {self.name!r} profile is not loaded.")
        matches = self._profile_data[self._profile_data["timestamp"] == self._ts_str(time_step)]
        if matches.empty:
            raise ValueError(f"DER {self.name!r} profile is missing timestamp {self._ts_str(time_step)}.")
        return matches.iloc[0]

    @staticmethod
    def _ts_str(time_step: datetime) -> str:
        """Normalize a timestamp to the string form stored in profile CSVs."""
        return pd.Timestamp(time_step).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _controller_fields(row: pd.Series) -> Dict[str, Any]:
        return {
            "controller_type": str(row["controller_type"]),
            "q_limit_kvar": float(row.get("q_limit_kvar", 0.0)),
            "v_ref_pu": float(row.get("v_ref_pu", 1.0)),
            "droop_kvar_per_pu": float(row.get("droop_kvar_per_pu", 0.0)),
        }

    def to_command(self, injection: Dict[str, Any]) -> DERCommand:
        """Convert model-specific profile data to a solver-neutral DER command."""
        parameters = {
            key: value
            for key, value in injection.items()
            if key not in {"p_kw", "q_kvar", "controller_type"}
        }
        if self.phases:
            parameters["terminal_nodes"] = self.phases
        if "soc" in parameters:
            parameters["soc_pct"] = parameters.pop("soc")
        return DERCommand(
            p_kw=float(injection["p_kw"]),
            q_kvar=float(injection["q_kvar"]),
            controller_type=str(injection.get("controller_type", "constant_pq")),
            parameters=parameters,
        )

    def advance_state(self, injection: Dict[str, Any], dt_hours: float) -> None:
        """Advance internal state after a converged QSTS time step."""

    def state_summary(self) -> Dict[str, float]:
        """Return state values to append to QSTS system results."""
        return {}


class PVModel(DERModel):
    """Photovoltaic generation model."""

    def __init__(self, name: str, profile_file: Optional[str | Path] = None, bus_name: Optional[str] = None, phases: Optional[list[int]] = None):
        super().__init__(name, profile_file, bus_name, phases)

    def _validate_profile(self) -> None:
        self._validate_controller_profile()
        if self._profile_data is not None:
            required = {"timestamp", "pv_kw"}
            missing = required.difference(self._profile_data.columns)
            if missing:
                raise ValueError(f"PV profile missing columns: {sorted(missing)}")

    def get_injection(self, time_step: datetime, dt_hours: float = 1.0) -> Dict[str, Any]:
        """Get PV generation for a specific timestamp.

        Returns:
            Dictionary with 'p_kw' (active power) and 'q_kvar' (reactive power, typically 0)
        """
        row = self._row_at(time_step)

        pv_kw = float(row["pv_kw"])
        q_kvar = float(row.get("pv_kvar", 0.0))

        return {"p_kw": pv_kw, "q_kvar": q_kvar, **self._controller_fields(row)}


class WindModel(DERModel):
    """Wind turbine generation model."""

    def __init__(self, name: str, profile_file: Optional[str | Path] = None, bus_name: Optional[str] = None, phases: Optional[list[int]] = None):
        super().__init__(name, profile_file, bus_name, phases)

    def _validate_profile(self) -> None:
        self._validate_controller_profile()
        if self._profile_data is not None:
            required = {"timestamp", "wind_kw"}
            missing = required.difference(self._profile_data.columns)
            if missing:
                raise ValueError(f"Wind profile missing columns: {sorted(missing)}")

    def get_injection(self, time_step: datetime, dt_hours: float = 1.0) -> Dict[str, Any]:
        """Get wind generation for a specific timestamp."""
        row = self._row_at(time_step)

        wind_kw = float(row["wind_kw"])
        q_kvar = float(row.get("wind_kvar", 0.0))

        return {"p_kw": wind_kw, "q_kvar": q_kvar, **self._controller_fields(row)}


class BESSModel(DERModel):
    """Battery Energy Storage System model."""

    def __init__(
        self,
        name: str,
        profile_file: Optional[str | Path] = None,
        bus_name: Optional[str] = None,
        phases: Optional[list[int]] = None,
        capacity_kwh: float = 200.0,
        p_rated_kw: float = 0.0,
        reserve_soc: float = 20.0,
        initial_soc: float = 50.0,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
    ):
        super().__init__(name, profile_file, bus_name, phases)
        self._soc = initial_soc
        self.capacity_kwh = capacity_kwh
        self.p_rated_kw = p_rated_kw
        self.reserve_soc = reserve_soc
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency

    def _validate_profile(self) -> None:
        self._validate_controller_profile()
        if self._profile_data is not None:
            required = {"timestamp", "bess_kw"}
            missing = required.difference(self._profile_data.columns)
            if missing:
                raise ValueError(f"BESS profile missing columns: {sorted(missing)}")

    def get_injection(self, time_step: datetime, dt_hours: float = 1.0) -> Dict[str, Any]:
        """Get BESS charge/discharge for a specific timestamp.

        The planned power is clipped so the SOC cannot break the reserve/100% bounds
        over exactly ``dt_hours``.

        Returns:
            Dictionary with 'p_kw' (positive=discharge, negative=charge), 'q_kvar', and 'soc'
        """
        row = self._row_at(time_step)

        bess_kw = float(row["bess_kw"])
        q_kvar = float(row.get("bess_kvar", 0.0))

        # 功率限制：不超过额定功率
        if self.p_rated_kw > 0:
            bess_kw = max(-self.p_rated_kw, min(self.p_rated_kw, bess_kw))

        if bess_kw > 0:
            bess_kw = min(
                bess_kw,
                (self._soc - self.reserve_soc) / 100.0 * self.capacity_kwh * self.discharge_efficiency / dt_hours,
            )
        elif bess_kw < 0:
            bess_kw = -min(
                abs(bess_kw),
                (100.0 - self._soc) / 100.0 * self.capacity_kwh / self.charge_efficiency / dt_hours,
            )

        return {"p_kw": bess_kw, "q_kvar": q_kvar, "soc": self._soc, **self._controller_fields(row)}

    def advance_state(self, injection: Dict[str, Any], dt_hours: float) -> None:
        p_kw = float(injection["p_kw"])
        if p_kw > 0:
            self._soc -= p_kw * dt_hours / self.discharge_efficiency / self.capacity_kwh * 100.0
        elif p_kw < 0:
            self._soc += abs(p_kw) * dt_hours * self.charge_efficiency / self.capacity_kwh * 100.0
        self._soc = max(self.reserve_soc, min(100.0, self._soc))

    def state_summary(self) -> Dict[str, float]:
        return {f"{self.name.lower()}_soc_pct": self._soc}


class EVModel(DERModel):
    """Electric Vehicle charging model."""

    def __init__(self, name: str, profile_file: Optional[str | Path] = None, bus_name: Optional[str] = None, phases: Optional[list[int]] = None):
        super().__init__(name, profile_file, bus_name, phases)

    def _validate_profile(self) -> None:
        self._validate_controller_profile()
        if self._profile_data is not None:
            required = {"timestamp", "ev_kw"}
            missing = required.difference(self._profile_data.columns)
            if missing:
                raise ValueError(f"EV profile missing columns: {sorted(missing)}")

    def get_injection(self, time_step: datetime, dt_hours: float = 1.0) -> Dict[str, Any]:
        """Get EV charging load for a specific timestamp.

        Returns:
            Dictionary with positive charging load ``p_kw`` and ``q_kvar``.
        """
        row = self._row_at(time_step)

        ev_kw = float(row["ev_kw"])
        q_kvar = float(row.get("ev_kvar", 0.0))

        return {"p_kw": ev_kw, "q_kvar": q_kvar, **self._controller_fields(row)}
