from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd


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
        element_type: str,
        profile_file: Optional[str | Path] = None,
        bus_name: Optional[str] = None,
    ):
        """Initialize DER model.

        Args:
            name: Name of the DER element in OpenDSS (e.g., "PV1", "Wind1")
            element_type: OpenDSS element type ("Generator", "Storage", "PVSystem", etc.)
            profile_file: Optional path to time-series profile CSV
        """
        self.name = name
        self.element_type = element_type
        self.profile_file = Path(profile_file) if profile_file else None
        self.bus_name = bus_name
        self._profile_data: Optional[pd.DataFrame] = None

    def load_profile(self) -> None:
        """Load time-series profile data from CSV file."""
        if self.profile_file and self.profile_file.exists():
            self._profile_data = pd.read_csv(self.profile_file)
            self._validate_profile()

    @abstractmethod
    def _validate_profile(self) -> None:
        """Validate that the loaded profile contains required columns."""
        pass

    @abstractmethod
    def get_injection(self, time_step: int) -> Dict[str, float]:
        """Get DER injection values for a specific time step.

        Args:
            time_step: Current simulation time step (e.g., hour)

        Returns:
            Dictionary with injection parameters, typically:
            - 'p_kw': Active power in kW (positive = generation, negative = consumption)
            - 'q_kvar': Reactive power in kvar
            - Additional parameters as needed (SOC for BESS, etc.)
        """
        pass

    @abstractmethod
    def apply_to_opendss(self, dss_interface: Any, injection: Dict[str, float]) -> None:
        """Apply injection values to OpenDSS model.

        Args:
            dss_interface: OpenDSS interface object (e.g., dss.ActiveCircuit)
            injection: Injection values from get_injection()
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this DER model."""
        return {
            "name": self.name,
            "element_type": self.element_type,
            "has_profile": self._profile_data is not None,
            "bus_name": self.bus_name,
        }

    def _validate_controller_profile(self) -> None:
        """Validate the controller type included in every DER profile."""
        if self._profile_data is None:
            return
        required = {"hour", "controller_type"}
        missing = required.difference(self._profile_data.columns)
        if missing:
            raise ValueError(f"DER profile missing columns: {sorted(missing)}")
        unsupported = set(self._profile_data["controller_type"].dropna().unique()).difference(
            self.SUPPORTED_CONTROLLERS
        )
        if unsupported:
            raise ValueError(f"Unsupported controller types: {sorted(unsupported)}")

    @staticmethod
    def _controller_fields(row: pd.Series) -> Dict[str, Any]:
        return {
            "controller_type": str(row["controller_type"]),
            "q_limit_kvar": float(row.get("q_limit_kvar", 0.0)),
            "v_ref_pu": float(row.get("v_ref_pu", 1.0)),
            "droop_kvar_per_pu": float(row.get("droop_kvar_per_pu", 0.0)),
        }

    def control_step(self, dss_interface: Any, injection: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate an algebraic DER controller after one OpenDSS power-flow solve."""
        if injection.get("controller_type") != "volt_var" or not self.bus_name:
            return injection

        dss_interface.SetActiveBus(self.bus_name)
        voltages = list(dss_interface.ActiveBus.puVmagAngle)[::2]
        if not voltages:
            return injection

        voltage_pu = sum(voltages) / len(voltages)
        q_limit = float(injection.get("q_limit_kvar", 0.0))
        q_command = float(injection.get("droop_kvar_per_pu", 0.0)) * (
            float(injection.get("v_ref_pu", 1.0)) - voltage_pu
        )
        updated = dict(injection)
        updated["q_kvar"] = max(-q_limit, min(q_limit, q_command))
        return updated

    def advance_state(self, injection: Dict[str, Any], dt_hours: float) -> None:
        """Advance internal state after a converged QSTS time step."""

    def state_summary(self) -> Dict[str, float]:
        """Return state values to append to QSTS system results."""
        return {}


class PVModel(DERModel):
    """Photovoltaic generation model."""

    def __init__(self, name: str = "PV1", profile_file: Optional[str | Path] = None, bus_name: Optional[str] = None):
        super().__init__(name, "Generator", profile_file, bus_name)

    def _validate_profile(self) -> None:
        self._validate_controller_profile()
        if self._profile_data is not None:
            required = {"hour", "pv_kw"}
            missing = required.difference(self._profile_data.columns)
            if missing:
                raise ValueError(f"PV profile missing columns: {sorted(missing)}")

    def get_injection(self, time_step: int) -> Dict[str, float]:
        """Get PV generation for a specific hour.

        Returns:
            Dictionary with 'p_kw' (active power) and 'q_kvar' (reactive power, typically 0)
        """
        if self._profile_data is None:
            return {"p_kw": 0.0, "q_kvar": 0.0}

        row = self._profile_data[self._profile_data["hour"] == time_step]
        if row.empty:
            return {"p_kw": 0.0, "q_kvar": 0.0}

        pv_kw = float(row.iloc[0]["pv_kw"])
        q_kvar = float(row.iloc[0].get("pv_kvar", 0.0))

        return {"p_kw": pv_kw, "q_kvar": q_kvar, **self._controller_fields(row.iloc[0])}

    def apply_to_opendss(self, dss_interface: Any, injection: Dict[str, float]) -> None:
        """Apply PV generation to OpenDSS Generator element."""
        generators = dss_interface.Generators

        if generators.Count == 0:
            if abs(injection["p_kw"]) > 1e-12:
                raise RuntimeError(
                    "The selected feeder has no Generator object. "
                    f"Cannot apply PV generation for {self.name}."
                )
            return

        names = [name.lower() for name in generators.AllNames]
        if self.name.lower() not in names:
            if abs(injection["p_kw"]) > 1e-12:
                raise RuntimeError(
                    f"Generator.{self.name} not found. "
                    f"Available generators: {names}"
                )
            return

        generators.Name = self.name
        generators.kW = float(injection["p_kw"])
        generators.kvar = float(injection["q_kvar"])


class WindModel(DERModel):
    """Wind turbine generation model."""

    def __init__(self, name: str = "Wind1", profile_file: Optional[str | Path] = None, bus_name: Optional[str] = None):
        super().__init__(name, "Generator", profile_file, bus_name)

    def _validate_profile(self) -> None:
        self._validate_controller_profile()
        if self._profile_data is not None:
            required = {"hour", "wind_kw"}
            missing = required.difference(self._profile_data.columns)
            if missing:
                raise ValueError(f"Wind profile missing columns: {sorted(missing)}")

    def get_injection(self, time_step: int) -> Dict[str, float]:
        """Get wind generation for a specific hour."""
        if self._profile_data is None:
            return {"p_kw": 0.0, "q_kvar": 0.0}

        row = self._profile_data[self._profile_data["hour"] == time_step]
        if row.empty:
            return {"p_kw": 0.0, "q_kvar": 0.0}

        wind_kw = float(row.iloc[0]["wind_kw"])
        q_kvar = float(row.iloc[0].get("wind_kvar", 0.0))

        return {"p_kw": wind_kw, "q_kvar": q_kvar, **self._controller_fields(row.iloc[0])}

    def apply_to_opendss(self, dss_interface: Any, injection: Dict[str, float]) -> None:
        """Apply wind generation to OpenDSS Generator element."""
        generators = dss_interface.Generators

        if generators.Count == 0 or self.name.lower() not in [n.lower() for n in generators.AllNames]:
            if abs(injection["p_kw"]) > 1e-12:
                raise RuntimeError(f"Generator.{self.name} not found for wind turbine.")
            return

        generators.Name = self.name
        generators.kW = float(injection["p_kw"])
        generators.kvar = float(injection["q_kvar"])


class BESSModel(DERModel):
    """Battery Energy Storage System model."""

    def __init__(self, name: str = "BESS1", profile_file: Optional[str | Path] = None, bus_name: Optional[str] = None):
        super().__init__(name, "Storage", profile_file, bus_name)
        self._soc = 50.0
        self.capacity_kwh = 200.0
        self.reserve_soc = 20.0
        self.charge_efficiency = 0.95
        self.discharge_efficiency = 0.95

    def _validate_profile(self) -> None:
        self._validate_controller_profile()
        if self._profile_data is not None:
            required = {"hour", "bess_kw"}
            missing = required.difference(self._profile_data.columns)
            if missing:
                raise ValueError(f"BESS profile missing columns: {sorted(missing)}")

    def get_injection(self, time_step: int) -> Dict[str, float]:
        """Get BESS charge/discharge for a specific hour.

        Returns:
            Dictionary with 'p_kw' (positive=discharge, negative=charge), 'q_kvar', and 'soc'
        """
        if self._profile_data is None:
            return {"p_kw": 0.0, "q_kvar": 0.0, "soc": self._soc}

        row = self._profile_data[self._profile_data["hour"] == time_step]
        if row.empty:
            return {"p_kw": 0.0, "q_kvar": 0.0, "soc": self._soc}

        bess_kw = float(row.iloc[0]["bess_kw"])
        q_kvar = float(row.iloc[0].get("bess_kvar", 0.0))

        if bess_kw > 0:
            bess_kw = min(
                bess_kw,
                (self._soc - self.reserve_soc) / 100.0 * self.capacity_kwh * self.discharge_efficiency,
            )
        elif bess_kw < 0:
            bess_kw = -min(
                abs(bess_kw),
                (100.0 - self._soc) / 100.0 * self.capacity_kwh / self.charge_efficiency,
            )

        return {"p_kw": bess_kw, "q_kvar": q_kvar, "soc": self._soc, **self._controller_fields(row.iloc[0])}

    def advance_state(self, injection: Dict[str, Any], dt_hours: float) -> None:
        p_kw = float(injection["p_kw"])
        if p_kw > 0:
            self._soc -= p_kw * dt_hours / self.discharge_efficiency / self.capacity_kwh * 100.0
        elif p_kw < 0:
            self._soc += abs(p_kw) * dt_hours * self.charge_efficiency / self.capacity_kwh * 100.0
        self._soc = max(self.reserve_soc, min(100.0, self._soc))

    def state_summary(self) -> Dict[str, float]:
        return {f"{self.name.lower()}_soc_pct": self._soc}

    def apply_to_opendss(self, dss_interface: Any, injection: Dict[str, float]) -> None:
        """Apply BESS operation to OpenDSS Storage element.

        Note: OpenDSS Storage elements use different property names than Generators.
        We need to use the DSS command interface to set kW for Storage elements.
        """
        from dss import dss

        storages = dss_interface.Storages

        if storages.Count == 0 or self.name.lower() not in [n.lower() for n in storages.AllNames]:
            if abs(injection["p_kw"]) > 1e-12:
                raise RuntimeError(f"Storage.{self.name} not found for BESS.")
            return

        # Set the active storage element
        storages.Name = self.name

        # Use DSS command interface to set Storage properties
        # Storage elements require State property to be set for charge/discharge
        p_kw = float(injection["p_kw"])
        q_kvar = float(injection["q_kvar"])

        if p_kw < 0:
            # Charging mode (negative power)
            dss(f"Storage.{self.name}.State=Charging")
            dss(f"Storage.{self.name}.kW={abs(p_kw)}")
        elif p_kw > 0:
            # Discharging mode (positive power)
            dss(f"Storage.{self.name}.State=Discharging")
            dss(f"Storage.{self.name}.kW={abs(p_kw)}")
        else:
            # Idle mode
            dss(f"Storage.{self.name}.State=Idling")
            dss(f"Storage.{self.name}.kW=0")

        # Set reactive power
        dss(f"Storage.{self.name}.kvar={q_kvar}")

        # Set state of charge if provided
        if "soc" in injection:
            dss(f"Storage.{self.name}.%stored={injection['soc']}")


class EVModel(DERModel):
    """Electric Vehicle charging model."""

    def __init__(self, name: str = "EV1", profile_file: Optional[str | Path] = None, bus_name: Optional[str] = None):
        super().__init__(name, "Load", profile_file, bus_name)

    def _validate_profile(self) -> None:
        self._validate_controller_profile()
        if self._profile_data is not None:
            required = {"hour", "ev_kw"}
            missing = required.difference(self._profile_data.columns)
            if missing:
                raise ValueError(f"EV profile missing columns: {sorted(missing)}")

    def get_injection(self, time_step: int) -> Dict[str, float]:
        """Get EV charging load for a specific hour.

        Returns:
            Dictionary with positive charging load ``p_kw`` and ``q_kvar``.
        """
        if self._profile_data is None:
            return {"p_kw": 0.0, "q_kvar": 0.0}

        row = self._profile_data[self._profile_data["hour"] == time_step]
        if row.empty:
            return {"p_kw": 0.0, "q_kvar": 0.0}

        ev_kw = float(row.iloc[0]["ev_kw"])
        q_kvar = float(row.iloc[0].get("ev_kvar", 0.0))

        return {"p_kw": ev_kw, "q_kvar": q_kvar, **self._controller_fields(row.iloc[0])}

    def apply_to_opendss(self, dss_interface: Any, injection: Dict[str, float]) -> None:
        """Apply EV charging to OpenDSS Load element."""
        loads = dss_interface.Loads

        names = [name.lower() for name in loads.AllNames]
        if self.name.lower() not in names:
            if abs(injection["p_kw"]) > 1e-12:
                raise RuntimeError(f"Load.{self.name} not found for EV charging.")
            return

        loads.Name = self.name
        loads.kW = float(injection["p_kw"])
        loads.kvar = float(injection["q_kvar"])
