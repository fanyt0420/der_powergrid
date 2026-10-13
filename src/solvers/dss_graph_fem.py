"""Graph P1/Galerkin power flow using DSS only to import element models.

Linear passive elements contribute their primitive admittance matrices.
Loads are solved independently as voltage-dependent nodal current injections.
The IEEE 13 example fixes the 650 boundary voltage and regulator taps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Mapping

import numpy as np
from dss import dss


@dataclass(frozen=True)
class ImportedLoad:
    name: str
    terminals: tuple[tuple[tuple[str, int], tuple[str, int] | None], ...]
    s_per_terminal_va: complex
    nominal_v: float
    model: int


class DssGraphFEMSolver:
    """Solve a fixed-topology feeder after importing its passive DSS Yprim data."""

    def __init__(self, master_file: str | Path, boundary_bus: str = "650",
                 fixed_taps: Mapping[str, float] | None = None,
                 excluded_transformers: tuple[str, ...] = ("Sub",)) -> None:
        self.master_file = Path(master_file).resolve()
        self.boundary_bus = boundary_bus.lower()
        self.fixed_taps = dict(fixed_taps or {})
        self.excluded_transformers = {name.lower() for name in excluded_transformers}

    def build(self) -> None:
        if not self.master_file.exists():
            raise FileNotFoundError(self.master_file)
        dss("Clear")
        dss(f'Redirect "{self.master_file}"')
        circuit = dss.ActiveCircuit
        # Use a specified, reproducible regulator configuration.
        for name, tap in self.fixed_taps.items():
            dss(f"Transformer.{name}.Taps=[1.0 {tap}]")
        dss("Set Controlmode=OFF")
        circuit.Solution.Solve()
        if not circuit.Solution.Converged:
            raise RuntimeError("Reference boundary calculation did not converge.")
        circuit.SetActiveBus(self.boundary_bus)
        bus = circuit.ActiveBus
        voltage_values = list(bus.Voltages)
        self.boundary = {(self.boundary_bus, int(ph)): complex(voltage_values[2 * k], voltage_values[2 * k + 1])
                         for k, ph in enumerate(bus.Nodes) if int(ph) != 0}
        if not self.boundary:
            raise ValueError("Boundary bus has no energized phases.")

        passive = []
        for category, names in (("Line", circuit.Lines.AllNames),
                                ("Transformer", circuit.Transformers.AllNames),
                                ("Capacitor", circuit.Capacitors.AllNames)):
            for name in names:
                if category == "Transformer" and name.lower() in self.excluded_transformers:
                    continue  # upstream transformer is outside the boundary
                circuit.SetActiveElement(f"{category}.{name}")
                element = circuit.ActiveCktElement
                nodes = []
                for busname, terminal in zip(element.BusNames,
                                             np.array(element.NodeOrder).reshape(element.NumTerminals, element.NumConductors)):
                    bus_id = busname.split(".")[0].lower()
                    nodes.extend((bus_id, int(ph)) if int(ph) != 0 else None for ph in terminal)
                raw = np.asarray(element.Yprim, dtype=float)
                size = len(nodes)
                if raw.size != 2 * size * size:
                    raise ValueError(f"Invalid Yprim size for {category}.{name}")
                admittance = (raw[0::2] + 1j * raw[1::2]).reshape(size, size)
                passive.append((f"{category}.{name}", nodes, admittance))

        node_set = set(self.boundary)
        for _, nodes, _ in passive:
            node_set.update(node for node in nodes if node is not None)
        node_set = {node for node in node_set if node[0] != "sourcebus"}
        self.nodes = sorted(node_set)
        self.index = {node: i for i, node in enumerate(self.nodes)}
        self.y = np.zeros((len(self.nodes), len(self.nodes)), dtype=complex)
        for _, nodes, admittance in passive:
            for i, left in enumerate(nodes):
                if left not in self.index:
                    continue
                for j, right in enumerate(nodes):
                    if right in self.index:
                        self.y[self.index[left], self.index[right]] += admittance[i, j]

        self.loads = []
        loads = circuit.Loads
        for name in loads.AllNames:
            loads.Name = name
            circuit.SetActiveElement(f"Load.{name}")
            element = circuit.ActiveCktElement
            bus_id = element.BusNames[0].split(".")[0].lower()
            phases = [int(ph) for ph in element.NodeOrder if int(ph) != 0]
            if loads.IsDelta:
                if len(phases) == 3:
                    pairs = [(phases[i], phases[(i + 1) % 3]) for i in range(3)]
                elif len(phases) == 2:
                    pairs = [(phases[0], phases[1])]
                else:
                    raise ValueError(f"Unsupported delta load: {name}")
                terminals = tuple(((bus_id, a), (bus_id, b)) for a, b in pairs)
                nominal = loads.kV * 1000
            else:
                terminals = tuple(((bus_id, ph), None) for ph in phases)
                nominal = loads.kV * 1000 / np.sqrt(3) if len(phases) == 3 else loads.kV * 1000
            self.loads.append(ImportedLoad(name, terminals,
                                            1000 * complex(loads.kW, loads.kvar) / len(terminals),
                                            nominal, int(loads.Model)))
        self.fixed = np.array([self.index[node] for node in self.boundary])
        self.free = np.array([i for i in range(len(self.nodes)) if i not in set(self.fixed)])
        self.vfixed = np.array([self.boundary[self.nodes[i]] for i in self.fixed])
        self.yff = self.y[np.ix_(self.free, self.free)]
        self.yfs = self.y[np.ix_(self.free, self.fixed)]
        np.linalg.solve(self.yff, np.zeros(len(self.free), dtype=complex))
        self.passive = passive
        self._build_paths(circuit)

    def _build_paths(self, circuit) -> None:
        """Partition the physical feeder into maximal paths with local x."""
        adjacency = defaultdict(list)
        edge_data = {}
        lines = circuit.Lines
        for name in lines.AllNames:
            lines.Name = name
            a, b = lines.Bus1.split(".")[0].lower(), lines.Bus2.split(".")[0].lower()
            units = str(lines.Units).lower()
            factor = 0.0003048 if "ft" in units else 1.609344 if "mi" in units else 1.0
            key = frozenset((a, b))
            edge_data.setdefault(key, {"length_km": float(lines.Length) * factor, "elements": []})
            edge_data[key]["elements"].append(f"Line.{name}")
        for name in circuit.Transformers.AllNames:
            if name.lower() in self.excluded_transformers:
                continue
            circuit.SetActiveElement(f"Transformer.{name}")
            buses = [item.split(".")[0].lower() for item in circuit.ActiveCktElement.BusNames]
            if len(buses) != 2:
                raise ValueError(f"Only two-winding transformers are supported: {name}")
            key = frozenset(buses)
            edge_data.setdefault(key, {"length_km": 0.0, "elements": []})
            edge_data[key]["elements"].append(f"Transformer.{name}")
        for key in edge_data:
            a, b = tuple(key)
            adjacency[a].append(b)
            adjacency[b].append(a)
        parent = {self.boundary_bus: None}
        order = [self.boundary_bus]
        for bus in order:
            for nxt in adjacency[bus]:
                if nxt == parent[bus]:
                    continue
                if nxt in parent:
                    raise ValueError("Feeder path graph contains a loop.")
                parent[nxt] = bus
                order.append(nxt)
        physical_buses = {bus for key in edge_data for bus in key}
        if set(parent) != physical_buses:
            raise ValueError("Physical feeder has disconnected buses.")
        children = defaultdict(list)
        for child, upstream in parent.items():
            if upstream is not None:
                children[upstream].append(child)
        starts = [self.boundary_bus] + [bus for bus in order if bus != self.boundary_bus and len(children[bus]) > 1]
        self.paths = []
        for start in starts:
            for child in children[start]:
                buses = [start, child]
                edges = [edge_data[frozenset((start, child))]]
                distances = [0.0, edges[0]["length_km"]]
                cursor = child
                while len(children[cursor]) == 1:
                    nxt = children[cursor][0]
                    edge = edge_data[frozenset((cursor, nxt))]
                    buses.append(nxt)
                    edges.append(edge)
                    distances.append(distances[-1] + edge["length_km"])
                    cursor = nxt
                self.paths.append({"id": f"path_{len(self.paths) + 1}", "buses": buses,
                                   "edges": edges, "x_km": distances})
        if sum(len(path["edges"]) for path in self.paths) != len(edge_data):
            raise RuntimeError("Path partition did not cover every physical connection.")

    def _load_currents(self, voltage: np.ndarray) -> np.ndarray:
        result = np.zeros(len(self.nodes), dtype=complex)
        for load in self.loads:
            for left, right in load.terminals:
                ia = self.index[left]
                ib = self.index[right] if right is not None else None
                across = voltage[ia] - (voltage[ib] if ib is not None else 0)
                if abs(across) < 1:
                    raise RuntimeError(f"Near-zero voltage at load {load.name}")
                if load.model == 1:  # constant power
                    current = np.conj(load.s_per_terminal_va / across)
                elif load.model == 2:  # constant impedance
                    current = np.conj(load.s_per_terminal_va) * across / load.nominal_v**2
                elif load.model == 5:  # constant current magnitude and power factor
                    current = np.conj(load.s_per_terminal_va) / load.nominal_v * across / abs(across)
                else:
                    raise ValueError(f"Unsupported DSS load model {load.model} on {load.name}")
                result[ia] -= current
                if ib is not None:
                    result[ib] += current
        return result

    def solve(self, tolerance: float = 1e-8, max_iterations: int = 200) -> dict:
        if not hasattr(self, "y"):
            raise RuntimeError("Call build() first")
        voltage = np.zeros(len(self.nodes), dtype=complex)
        for node, i in self.index.items():
            bus, ph = node
            circuit = dss.ActiveCircuit
            circuit.SetActiveBus(bus)
            kvbase = circuit.ActiveBus.kVBase
            voltage[i] = kvbase * 1000 * np.exp(1j * np.deg2rad({1: 0, 2: -120, 3: 120}[ph]))
        voltage[self.fixed] = self.vfixed
        for iteration in range(1, max_iterations + 1):
            injection = self._load_currents(voltage)
            updated = voltage.copy()
            updated[self.free] = np.linalg.solve(self.yff,
                                                 injection[self.free] - self.yfs @ self.vfixed)
            change = float(np.max(np.abs(updated - voltage)))
            voltage = updated
            if change < tolerance:
                break
        else:
            raise RuntimeError("Graph FEM load-flow iteration did not converge")
        residual = self.y @ voltage - self._load_currents(voltage)
        self.voltage = voltage
        self.element_terminal_currents = {}
        for name, nodes, admittance in self.passive:
            local_v = np.array([voltage[self.index[node]] if node in self.index else 0j for node in nodes])
            local_i = admittance @ local_v
            currents = defaultdict(complex)
            for node, current in zip(nodes, local_i):
                if node is not None:
                    currents[node] += current
            self.element_terminal_currents[name.lower()] = currents
        return {"iterations": iteration, "max_residual_a": float(np.max(np.abs(residual[self.free]))),
                "voltages": {f"{bus}.{ph}": complex(voltage[i]) for (bus, ph), i in self.index.items()}}

    def reference_voltages(self) -> dict[str, complex]:
        circuit = dss.ActiveCircuit
        reference = {}
        for bus in circuit.AllBusNames:
            circuit.SetActiveBus(bus)
            active = circuit.ActiveBus
            values = list(active.Voltages)
            for k, ph in enumerate(active.Nodes):
                if int(ph):
                    reference[f"{bus.lower()}.{int(ph)}"] = complex(values[2 * k], values[2 * k + 1])
        return reference

    def path_profile_records(self) -> list[dict]:
        if not hasattr(self, "voltage"):
            raise RuntimeError("Call solve() before requesting path profiles.")
        circuit = dss.ActiveCircuit
        rows = []
        for path in self.paths:
            for k, (bus, x) in enumerate(zip(path["buses"], path["x_km"])):
                circuit.SetActiveBus(bus)
                kvbase = float(circuit.ActiveBus.kVBase)
                outgoing = "+".join(path["edges"][k]["elements"]) if k < len(path["edges"]) else ""
                for ph in (1, 2, 3):
                    node = (bus, ph)
                    if node not in self.index:
                        continue
                    value = self.voltage[self.index[node]]
                    branch_current = (sum((self.element_terminal_currents[name.lower()].get(node, 0j)
                                           for name in path["edges"][k]["elements"]), 0j)
                                      if k < len(path["edges"]) else None)
                    rows.append({"path": path["id"], "bus": bus, "x_km": x, "phase": ph,
                                 "v_volts": abs(value), "v_pu": abs(value) / (kvbase * 1000),
                                 "angle_deg": float(np.rad2deg(np.angle(value))),
                                 "outgoing_elements": outgoing,
                                 "outgoing_current_a": abs(branch_current) if branch_current is not None else None})
        return rows
