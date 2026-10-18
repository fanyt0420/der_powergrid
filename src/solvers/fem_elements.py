"""Native phase-domain element matrices. No OpenDSS runtime is used here.

Lines use P1 series stiffness with lumped pi charging; transformers use winding
connection maps and a leakage element. All voltages are phasors in volts,
admittances in siemens, and powers in VA. See doc/fem_solver.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

KM_PER_UNIT = {"km": 1., "mi": 1.609344, "ft": .0003048, "m": .001,
               "kft": .3048, "cm": .00001, "in": .0000254}


@dataclass
class Element:
    data: Mapping[str, Any]
    nodes: np.ndarray
    y: np.ndarray
    nleft: int
    length_km: float = 0.


def connection_map(phases: list[int], connection: str, *, previous: bool = False) -> np.ndarray:
    """Map grounded phase voltages to individual wye/delta coil voltages."""
    n = len(phases)
    if connection == "wye":
        return np.eye(n)
    if connection != "delta" or n not in (2, 3):
        raise NotImplementedError("Only grounded wye and two/three-terminal delta connections are supported.")
    if n == 2:
        return np.array([[1., -1.]])
    result = np.eye(n)
    for i in range(n):
        result[i, (i - 1 if previous else i + 1) % n] -= 1.
    return result


def coil_voltage(item: Mapping[str, Any], nphases: int, nominal_ll_kv: float) -> float:
    if "kv_ln" in item:
        return float(item["kv_ln"]) * 1000
    voltage = float(item.get("kv", item.get("kv_ll", nominal_ll_kv))) * 1000
    if str(item.get("connection", "wye")).lower() == "wye":
        # Explicit kV on a single-phase device is its phase-to-neutral rating.
        if nphases > 1 or ("kv" not in item and "kv_ll" not in item):
            voltage /= np.sqrt(3)
    return voltage


def source_impedance(source: Mapping[str, Any]) -> np.ndarray | None:
    """Sequence impedance from explicit ohms or three/one-phase fault MVA."""
    if "r1" in source or "x1" in source:
        z1 = complex(source.get("r1", 0), source.get("x1", 0))
        z0 = complex(source.get("r0", z1.real), source.get("x0", z1.imag))
    elif "mvasc3" in source:
        z1mag = float(source["base_kv_ll"]) ** 2 / float(source["mvasc3"])
        xr1, xr0 = float(source.get("x1r1", 4)), float(source.get("x0r0", 3))
        z1 = z1mag * complex(1, xr1) / np.hypot(1, xr1)
        if "mvasc1" in source:
            target = 3 * float(source["base_kv_ll"]) ** 2 / float(source["mvasc1"])
            a, b = 1 + xr0 ** 2, 4 * (z1.real + xr0 * z1.imag)
            c = 4 * abs(z1) ** 2 - target ** 2
            if b * b - 4 * a * c < 0:
                raise ValueError("Inconsistent source short-circuit ratings.")
            r0 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
            if r0 <= 0:
                raise ValueError("Source ratings imply a non-positive zero-sequence resistance.")
            z0 = r0 * complex(1, xr0)
        else:
            z0 = z1
    else:
        return None
    if abs(z1) == 0 or abs(z0) == 0:
        raise ValueError("Source sequence impedances must be nonzero.")
    return np.eye(3) * z1 + np.ones((3, 3)) * (z0 - z1) / 3


def make_branch(branch, network, index, tap_pu: float = 1.) -> Element:
    kind = str(branch["kind"])
    left_ph = list(branch.get("from_terminal_nodes", branch["phases"]))
    right_ph = list(branch.get("to_terminal_nodes", branch["phases"]))
    left = [index[(str(branch["from_bus"]).lower(), p)] for p in left_ph]
    right = [index[(str(branch["to_bus"]).lower(), p)] for p in right_ph]
    n = len(left)
    nodes = np.array(left + right)
    length_km = 0.
    if kind in {"line", "switch"}:
        if kind == "switch":
            if not branch.get("normally_closed", True):
                return Element(branch, nodes, np.zeros((2*n, 2*n), complex), n)
            z1 = complex(branch.get("r1_ohm", .0001), branch.get("x1_ohm", 0))
            z0 = complex(branch.get("r0_ohm", z1.real), branch.get("x0_ohm", z1.imag))
            z = np.eye(n) * z1 + np.ones((n, n)) * (z0 - z1) / 3
            shunt = np.zeros((n, n), complex)
        else:
            code = network.line_codes[str(branch["line_code"])] if "line_code" in branch else None
            unit = str(code.get("unit", "km")).lower() if code else "km"
            length_unit = str(branch.get("length_unit", unit if "length" in branch else "km")).lower()
            if unit not in KM_PER_UNIT or length_unit not in KM_PER_UNIT:
                raise NotImplementedError(f"Unsupported line unit in {branch['id']}.")
            length_km = float(branch.get("length", branch.get("length_km"))) * KM_PER_UNIT[length_unit]
            if code:
                z = (np.array(code["r_ohm_per_unit"]) + 1j*np.array(code["x_ohm_per_unit"])) * length_km / KM_PER_UNIT[unit]
                capacitance = np.asarray(code.get("c_nf_per_unit", np.zeros((n, n)))) * length_km / KM_PER_UNIT[unit]
            else:
                z = np.array([[complex(*v) for v in row] for row in branch["z_ohm_per_km"]]) * length_km
                capacitance = np.asarray(branch.get("c_nf_per_km", np.zeros((n, n)))) * length_km
            shunt = 1j * 2*np.pi * float(network.base["frequency_hz"]) * capacitance * 1e-9 / 2
        series = np.linalg.inv(z)
        y = np.block([[series + shunt, -series], [-series, series + shunt]])
    elif kind in {"transformer", "regulator"}:
        windings = branch["windings"]
        if len(windings) != 2:
            raise NotImplementedError("FEM currently supports two-winding transformers only.")
        if str(branch.get("lead_lag", "lag")).lower() != "lag":
            raise NotImplementedError("Transformer phase displacement currently supports lead_lag=lag only.")
        maps, bases = [], []
        for w in windings:
            ph = list(w.get("terminal_nodes", branch["phases"]))
            if len(ph) != n or any(float(w.get(key, 0)) != 0 for key in ("rneut", "xneut")):
                raise NotImplementedError("Transformer windings must have matching dimensions and grounded neutrals.")
            conn = str(w["connection"]).lower()
            maps.append(connection_map(ph, conn, previous=str(windings[0]["connection"]).lower() == "delta"))
            bases.append(coil_voltage(w, n, 0))
        if maps[0].shape[0] != maps[1].shape[0]:
            raise NotImplementedError("Transformer coil counts must match.")
        ncoil = maps[0].shape[0]
        s_phase = float(windings[0]["kva"]) * 1000 / ncoil
        r_pct = sum(float(w.get("resistance_percent", float(branch.get("load_loss_percent", 0)) / 2)) for w in windings)
        zpu = complex(r_pct, float(branch.get("xhl_percent", branch.get("xhl_pct", 0)))) / 100
        if abs(zpu) == 0:
            raise ValueError(f"Zero transformer leakage impedance in {branch['id']}.")
        tap1 = float(windings[0].get("tap_pu", 1))
        tap2 = float(windings[1].get("tap_pu", 1)) * tap_pu
        b = np.hstack((maps[0] / (bases[0]*tap1), -maps[1] / (bases[1]*tap2)))
        y = (s_phase / zpu) * (b.T @ b)
        # Small physical/numerical magnetizing shunt, referred to each winding.
        ppm = float(branch.get("ppm_antifloat", 1)) * 1e-6
        for k in range(2):
            sl = slice(k*n, (k+1)*n)
            y[sl, sl] += -1j * ppm * s_phase / bases[k]**2 * np.eye(n)
        no_load = complex(float(branch.get("no_load_loss_percent", 0)), -float(branch.get("imag_percent", 0))) / 100
        y[:n, :n] += no_load * s_phase / bases[0]**2 * (maps[0].T @ maps[0])
    else:
        raise NotImplementedError(f"Unsupported branch kind {kind!r}.")
    if not branch.get("in_service", True):
        y = np.zeros_like(y)
    return Element(branch, nodes, y, n, length_km)
