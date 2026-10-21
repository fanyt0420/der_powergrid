"""Frozen dynamic network using the repository's phase-domain element builders."""

from dataclasses import dataclass
import numpy as np

from src.solvers.fem_elements import make_branch, connection_map, coil_voltage


PHASE_ANGLE = {1: 0.0, 2: -2 * np.pi / 3, 3: 2 * np.pi / 3}


@dataclass
class Element:
    name: str
    indices: np.ndarray
    y: np.ndarray


class PassiveNetwork:
    def __init__(self, network, nodes, taps, config):
        self.nodes = tuple(str(node).lower() for node in nodes)
        self.index = {node: i for i, node in enumerate(self.nodes)}
        self.bus_by_name = {b["id"].lower(): b for b in network.buses}
        self.vbase = np.array([self.bus_by_name[n.rsplit(".", 1)[0]]["nominal_kv_ll"] * 1000 / np.sqrt(3) for n in self.nodes])
        self.ibase = config["system_base_va"] / 3 / self.vbase
        slack = network.base["slack_bus"].lower()
        self.fixed = np.array([i for i, n in enumerate(self.nodes) if n.rsplit(".", 1)[0] == slack])
        self.free = np.array([i for i in range(len(nodes)) if i not in self.fixed])
        self.elements = []
        self.y = np.zeros((len(nodes), len(nodes)), complex)
        index = {(n.rsplit(".", 1)[0], int(n.rsplit(".", 1)[1])): i for i, n in enumerate(self.nodes)}
        for original in network.branches:
            if not original.get("in_service", True):
                continue
            branch = dict(original)
            if branch["kind"] == "switch":
                if not branch.get("normally_closed", True):
                    continue
                # Match the actual DSS length of the snapshot without changing
                # the shared builder or the input network's physical parameters.
                for key in ("r1_ohm", "x1_ohm", "r0_ohm", "x0_ohm"):
                    if key in branch:
                        branch[key] *= config["switch_impedance_multiplier"]
            element = make_branch(branch, network, index, taps.get(branch["id"].lower(), 1.0))
            category = "Line" if branch["kind"] in {"line", "switch"} else "Transformer"
            self.add(category + "." + branch["id"], element.nodes, element.y)
        for capacitor in network.shunt_devices:
            if not capacitor.get("in_service", True):
                continue
            phases = capacitor.get("terminal_nodes", capacitor["phases"])
            ids = self.indices(capacitor["bus"], phases)
            c = connection_map(phases, capacitor["connection"])
            nominal = self.bus_by_name[capacitor["bus"].lower()]["nominal_kv_ll"]
            base = coil_voltage(capacitor, len(phases), nominal)
            y = 1j * capacitor["q_kvar"] * 1000 / len(c) / base**2 * (c.T @ c)
            self.add("Capacitor." + capacitor["id"], ids, y)

    def indices(self, bus, phases):
        return np.array([self.index[f"{bus.lower()}.{p}"] for p in phases])

    def add(self, name, indices, y):
        self.elements.append(Element(name, indices, y))
        self.y[np.ix_(indices, indices)] += y

    def current(self, voltage):
        # Accumulate per element to reduce cancellation near the closed switch.
        result = np.zeros(len(voltage), np.clongdouble)
        for element in self.elements:
            result[element.indices] += element.y.astype(np.clongdouble) @ voltage[element.indices]
        return result

    def losses_va(self, voltage):
        return sum(np.sum(v * np.conj(e.y.astype(np.clongdouble) @ v))
                   for e in self.elements for v in [voltage[e.indices].astype(np.clongdouble)])

    def connection(self, device):
        phases = device.get("terminal_nodes", device["phases"])
        indices = self.indices(device["bus"], phases)
        result = np.zeros((len(indices), len(self.nodes)))
        result[:, indices] = np.eye(len(indices))
        return connection_map(phases, device.get("connection", "wye")) @ result
