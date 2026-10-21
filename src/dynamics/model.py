"""Instance-based GFM/GFL controller states coupled to a three-phase network."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.dynamics.network import PHASE_ANGLE
from src.solvers.fem_elements import coil_voltage


def pack(values):
    return np.r_[values.real, values.imag]


def unpack(values):
    n = len(values) // 2
    return values[:n] + 1j * values[n:]


def validate_dynamic_devices(config, network):
    """Validate instance interfaces before allocating states or reading a snapshot."""
    by_id = {device["id"]: device for device in network.devices}
    seen = set()
    for item in config["dynamic_devices"]:
        name = item["device_id"]
        if name in seen:
            raise ValueError(f"Duplicate dynamic device ID: {name}")
        seen.add(name)
        device = by_id.get(name)
        if device is None or device["kind"] in {"load", "ev"}:
            raise ValueError(f"Invalid dynamic device: {name}")
        if not device.get("in_service", True):
            raise ValueError(f"Dynamic device is out of service: {name}")
        if device.get("connection", "wye") != "wye":
            raise ValueError(f"Dynamic device requires grounded-wye ports: {name}")
        model = item.get("model")
        if model not in {"inertia", "droop", "gfl"}:
            raise ValueError(f"Unsupported controller model {model!r} for {name}; supported: inertia, droop, gfl")
        positive = ["s_ref_va", "tau_p_s", "tau_q_s", "current_limit_pu"]
        if model == "gfl":
            if len(device.get("terminal_nodes", device["phases"])) not in {1, 3}:
                raise ValueError(f"GFL requires single-phase or three-phase ports: {name}")
            positive += ["tau_i_s", "pll_kp_hz", "pll_ki_hz_per_s"]
            nonnegative = []
        else:
            positive += ["tau_e_s", "x_c_pu"] + (["H_s"] if model == "inertia" else ["d_p"])
            nonnegative = ["r_c_pu", "n_q"] + (["D", "k_pf"] if model == "inertia" else [])
        for key in positive + nonnegative:
            value = item.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value < 0 or (key in positive and value == 0)):
                raise ValueError(f"Invalid parameter {key}={value!r} for {name}")


@dataclass
class Port:
    device: dict
    c: np.ndarray
    power_va: complex
    nominal_v: float
    exponent: int


class KuramotoDAE:
    def __init__(self, network, passive, snapshot, config):
        validate_dynamic_devices(config, network)
        self.network, self.passive, self.snapshot, self.config = network, passive, snapshot, config
        self.frequency_hz = network.base["frequency_hz"]
        self.dynamic = []
        by_id = {d["id"]: d for d in network.devices}
        dynamic_ids = {c["device_id"] for c in config["dynamic_devices"]}
        self.ports = []
        for device in network.devices:
            if not device.get("in_service", True) or device["id"] in dynamic_ids:
                continue
            c = passive.connection(device)
            load = device["kind"] == "load"
            s = snapshot["loads_va" if load else "der_va"][device["id"]]
            if not load and device["kind"] != "ev":
                s = -s  # Port powers use consumption-positive convention.
            exponent = {1: 0, 2: 2, 5: 1}[device.get("load_model", 1)] if load else 0
            nominal = coil_voltage(device, len(c), passive.bus_by_name[device["bus"].lower()]["nominal_kv_ll"])
            self.ports.append(Port(dict(device), c, s, nominal, exponent))
        for name, s in snapshot.get("auxiliary_consumption_va", {}).items():
            if abs(s) > 1e-6:
                device = dict(by_id[name], id=name + "_auxiliary", kind="auxiliary")
                device["parent_device_id"] = name
                c = passive.connection(device)
                nominal = passive.vbase[passive.indices(device["bus"], device["phases"])][0]
                self.ports.append(Port(device, c, s, nominal, 0))
        self.state_names = []
        for config_device in config["dynamic_devices"]:
            d = dict(config_device)
            device = by_id[d["device_id"]]
            if device.get("connection", "wye") != "wye":
                raise ValueError("Dynamic devices require grounded-wye ports")
            d["c"] = passive.connection(device)
            d["phases"] = tuple(device.get("terminal_nodes", device["phases"]))
            d["v_nom"] = passive.vbase[passive.indices(device["bus"], d["phases"])][0]
            n = len(d["phases"])
            d["i_ref"] = d["s_ref_va"] / n / d["v_nom"]
            d["z_ref"] = n * d["v_nom"]**2 / d["s_ref_va"]
            if d["model"] != "gfl":
                d["z_c"] = d["z_ref"] * complex(d["r_c_pu"], d["x_c_pu"])
            d["phase_rotation"] = np.exp(1j * np.array([PHASE_ANGLE[p] for p in d["phases"]]))
            s = snapshot["der_va"][d["device_id"]] / d["s_ref_va"]
            d["p_star"], d["q_star"] = s.real, s.imag
            suffixes = ["delta_rad", "delta_f_hz", "e_pu", "p_m_pu", "q_m_pu"] if d["model"] == "inertia" else ["delta_rad", "e_pu", "p_m_pu", "q_m_pu"]
            if d["model"] == "gfl":
                suffixes = ["theta_pll_rad", "xi_pll_s", "p_f_pu", "q_f_pu", "i_d_pu", "i_q_pu"]
            d["slice"] = slice(len(self.state_names), len(self.state_names) + len(suffixes))
            self.state_names += [d["device_id"] + "_" + s for s in suffixes]
            self.dynamic.append(d)
        self.nx = len(self.state_names)
        self.ny = 2 * len(passive.free)
        self.fixed_voltage = snapshot["voltage"][passive.fixed].copy()
        if config["source_boundary"] == "nominal_ideal":
            self.fixed_voltage = np.array([passive.vbase[i] * network.source.get("voltage_pu", 1.0) * np.exp(1j * (np.deg2rad(network.source.get("angle_deg", 0.0)) + PHASE_ANGLE[int(passive.nodes[i].rsplit('.', 1)[1])])) for i in passive.fixed])
        elif config["source_boundary"] != "frozen_qsts_terminal":
            raise ValueError("Unknown source_boundary")
        # Exact linear network contribution in voltage-pu coordinates.
        yu = passive.y[np.ix_(passive.free, passive.free)] * passive.vbase[passive.free][None, :] / passive.ibase[passive.free][:, None]
        self.g_linear = np.block([[yu.real, -yu.imag], [yu.imag, yu.real]])

    def voltage(self, y):
        v = np.empty(len(self.passive.nodes), np.clongdouble)
        v[self.passive.fixed] = self.fixed_voltage
        v[self.passive.free] = unpack(y) * self.passive.vbase[self.passive.free]
        return v

    def voltage_coordinates(self, v):
        return pack(v[self.passive.free] / self.passive.vbase[self.passive.free])

    @staticmethod
    def gfl_voltage(d, u):
        # Mean after phase removal equals positive sequence for ABC ports.
        reference = np.mean(u / d["phase_rotation"]) / d["v_nom"]
        if abs(reference) < .05:
            raise ValueError(f"GFL reference voltage below 0.05 pu: {d['device_id']}")
        return reference

    def initialize_gfl(self, x, y):
        """Recover locked controller equilibrium from the current network voltage."""
        v = self.voltage(y)
        for d in self.dynamic:
            if d["model"] == "gfl":
                reference = self.gfl_voltage(d, d["c"] @ v)
                x[d["slice"]] = [np.angle(reference), 0, d["p_star"], d["q_star"],
                                  d["p_star"] / abs(reference), -d["q_star"] / abs(reference)]

    def evaluate(self, x, y, *, scales=None, tripped=frozenset(), all_pq=False):
        v = self.voltage(y)
        injection = np.zeros(len(v), np.clongdouble)
        f = np.zeros(self.nx)
        records = []
        load_power = 0j
        static_power = 0j
        for port in self.ports:
            name = port.device["id"]
            if name in tripped or port.device.get("parent_device_id") in tripped:
                records.append({"device_id": name, "connected": False, "p_kw": 0.0, "q_kvar": 0.0, "current_max_a": 0.0})
                continue
            u = port.c @ v
            if np.min(np.abs(u)) < 0.05 * port.nominal_v:
                raise ValueError(f"Voltage outside phasor model domain: {name}")
            factor = (scales or {}).get(name, (1.0, 1.0))
            s = complex(port.power_va.real * factor[0], port.power_va.imag * factor[1]) / len(u) * (np.abs(u) / port.nominal_v)**port.exponent
            current = np.conj(s / u)
            injection -= port.c.T @ current
            total = np.sum(s)
            if port.device["kind"] in {"load", "auxiliary"}:
                load_power += total
            else:
                static_power -= total
                records.append({"device_id": name, "connected": True, "p_kw": -total.real / 1000, "q_kvar": -total.imag / 1000, "current_max_a": float(max(abs(current)))})
        for d in self.dynamic:
            state = x[d["slice"]]
            if d["model"] == "gfl":
                theta, xi, pf, qf, id_, iq = state
                u = d["c"] @ v
                connected = d["device_id"] not in tripped
                df = 0.0
                if not connected:
                    current = np.zeros(len(u), complex)
                elif all_pq:
                    current = np.conj(self.snapshot["der_va"][d["device_id"]] / len(u) / u)
                else:
                    reference = self.gfl_voltage(d, u)
                    error = np.sin(np.angle(reference) - theta)
                    df = d["pll_kp_hz"] * error + d["pll_ki_hz_per_s"] * xi
                    udq = reference * np.exp(-1j * theta)
                    target = complex(pf, -qf) / np.conj(udq)
                    current = d["i_ref"] * complex(id_, iq) * np.exp(1j * theta) * d["phase_rotation"]
                    f[d["slice"]] = [2 * np.pi * df, error,
                        (d["p_star"] - pf) / d["tau_p_s"], (d["q_star"] - qf) / d["tau_q_s"],
                        (target.real - id_) / d["tau_i_s"], (target.imag - iq) / d["tau_i_s"]]
                injection += d["c"].T @ current
                power = np.sum(u * np.conj(current))
                gamma = np.abs(current) / d["i_ref"]
                worst = int(np.argmax(gamma))
                records.append({"device_id": d["device_id"], "model": "gfl", "connected": connected,
                    "p_kw": power.real / 1000, "q_kvar": power.imag / 1000,
                    "current_max_a": float(abs(current[worst])), "current_ratio": float(gamma[worst]),
                    "worst_phase": d["phases"][worst], "current_limit_pu": d["current_limit_pu"],
                    "delta_f_hz": float(df) if connected else float("nan")})
                continue
            delta = state[0]
            offset = 1 if d["model"] == "inertia" else 0
            e, pm, qm = state[1 + offset:4 + offset]
            u = d["c"] @ v
            connected = d["device_id"] not in tripped
            if not connected:
                current = np.zeros(len(u), complex)
            elif all_pq:
                current = np.conj(self.snapshot["der_va"][d["device_id"]] / len(u) / u)
            else:
                internal = d["v_nom"] * e * np.exp(1j * (delta + np.array([PHASE_ANGLE[p] for p in d["phases"]])))
                current = (internal - u) / d["z_c"]
            injection += d["c"].T @ current
            power = np.sum(u * np.conj(current)) / d["s_ref_va"]
            df = state[1] if offset else (d["p_star"] - pm) / d["d_p"]
            if connected:
                dx = [2 * np.pi * df]
                if offset:
                    dx += [(d["p_star"] - pm - (d["D"] + d["k_pf"]) * df) / (2 * d["H_s"] / self.frequency_hz)]
                dx += [(d.get("e_star", e) - e - d["n_q"] * (qm - d["q_star"])) / d["tau_e_s"], (power.real - pm) / d["tau_p_s"], (power.imag - qm) / d["tau_q_s"]]
                f[d["slice"]] = dx
            gamma = np.abs(current) / d["i_ref"]
            worst = int(np.argmax(gamma))
            records.append({"device_id": d["device_id"], "connected": connected, "p_kw": power.real * d["s_ref_va"] / 1000, "q_kvar": power.imag * d["s_ref_va"] / 1000,
                            "current_max_a": float(abs(current[worst])), "current_ratio": float(gamma[worst]), "worst_phase": d["phases"][worst],
                            "current_limit_pu": d["current_limit_pu"], "delta_f_hz": float(df) if connected else float("nan")})
        return f, injection, records, load_power, static_power

    def kcl(self, x, y, **kwargs):
        injection = self.evaluate(x, y, **kwargs)[1]
        return pack((self.passive.current(self.voltage(y)) - injection)[self.passive.free] / self.passive.ibase[self.passive.free])

    def initial_guess(self):
        x = np.zeros(self.nx)
        v = self.snapshot["voltage"]
        for d in self.dynamic:
            if d["model"] == "gfl":
                continue
            u = d["c"] @ v
            current = np.conj(self.snapshot["der_va"][d["device_id"]] / len(u) / u)
            ephase = u + d["z_c"] * current
            common = np.mean(ephase * np.exp(-1j * np.array([PHASE_ANGLE[p] for p in d["phases"]])))
            values = [np.angle(common)]
            if d["model"] == "inertia":
                values += [0.0]
            values += [abs(common) / d["v_nom"], d["p_star"], d["q_star"]]
            x[d["slice"]] = values
        y = self.voltage_coordinates(v)
        self.initialize_gfl(x, y)
        return x, y

    def domain_violations(self, records):
        return [r for r in records if r.get("current_ratio", 0) > r.get("current_limit_pu", float("inf"))]
