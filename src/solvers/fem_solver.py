"""Native graph P1/phase-domain QSTS solver; see doc/fem_solver.md.

No DSS file, primitive matrix or voltage solution is used by this backend.
This equivalent-circuit FEM is not a spatial PDE or a demonstrated speed-up.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from src.power_flow import NetworkModel, OperatingPoint, PFResult, PFState, SolverContext
from src.solvers.fem_elements import Element, coil_voltage, connection_map, make_branch, source_impedance


class FiniteElementPFSolver:
    """Grounded phase-domain feeder solver, including the bundled DER cases."""

    def __init__(self, *, voltage_tolerance_pu=1e-9, max_iterations=200,
                 regulator_mode="auto", max_regulator_iterations=100):
        if regulator_mode not in {"auto", "fixed"}:
            raise ValueError("regulator_mode must be auto or fixed")
        self.voltage_tolerance_pu = voltage_tolerance_pu
        self.max_iterations = max_iterations
        self.regulator_mode = regulator_mode
        self.max_regulator_iterations = max_regulator_iterations
        self.network = None

    def build(self, network: NetworkModel) -> None:
        self.network = network
        self.keys = tuple((str(b["id"]).lower(), int(p)) for b in network.buses for p in b["phases"])
        self.index = {key: i for i, key in enumerate(self.keys)}
        self.nominal_v = np.array([float(b["nominal_kv_ll"])*1000/np.sqrt(3) for b in network.buses for _ in b["phases"]])
        self.bus_by_name = {str(b["id"]).lower(): b for b in network.buses}
        slack = str(network.base["slack_bus"]).lower()
        self.source_indices = np.array([self.index[(slack, p)] for p in (1, 2, 3)])
        angle = np.deg2rad(float(network.source.get("angle_deg", 0)))
        vbase = float(network.source.get("base_kv_ll", self.bus_by_name[slack]["nominal_kv_ll"])) * 1000/np.sqrt(3)
        self.emf = vbase * float(network.source.get("voltage_pu", 1)) * np.exp(1j*(angle+np.deg2rad([0, -120, 120])))
        zsource = source_impedance(network.source)
        self.source_y = np.linalg.inv(zsource) if zsource is not None else None
        self.fixed = self.source_indices if zsource is None else np.array([], dtype=int)
        self.free = np.setdiff1d(np.arange(len(self.keys)), self.fixed)
        self.norton = np.zeros(len(self.keys), complex)
        if self.source_y is not None:
            self.norton[self.source_indices] = self.source_y @ self.emf
        self.tap_positions, self.regulators = {}, {}
        for branch in network.branches:
            if branch["kind"] != "regulator":
                continue
            name = str(branch["id"])
            if "tap" not in branch:
                raise ValueError(f"Missing tap range for regulator {name}.")
            if branch.get("control", {}).get("controlled_winding", 2) != 2 or len(branch["phases"]) != 1:
                raise NotImplementedError("Regulation supports single-phase, secondary-controlled regulators.")
            self.tap_positions[name] = int(branch["tap"].get("position", 0))
            self.regulators[name] = branch
        self.device_models = {}
        for device in network.devices:
            kind = str(device["kind"])
            if kind not in {"load", "pv", "wind", "ev", "bess"}:
                raise NotImplementedError(f"Unsupported device {kind}.")
            model = int(device.get("load_model", 1)) if kind in {"load", "ev"} else 1
            if model not in {1, 2, 5}:
                raise NotImplementedError(f"Unsupported load model {model} on {device['id']}.")
            ids, c, vbase = self._terminal_model(device)
            self.device_models[str(device["id"]).lower()] = (device, ids, c, vbase, model)
        self.shunts = []
        for cap in network.shunt_devices:
            if cap["kind"] != "capacitor":
                raise NotImplementedError("Only fixed capacitor shunts are supported.")
            if cap.get("in_service", True):
                ids, c, vbase = self._terminal_model(cap)
                self.shunts.append((ids, 1j*float(cap["q_kvar"])*1000/len(c)/vbase**2 * (c.T @ c)))
        self.factorizations = 0
        self._assemble()
        self._build_paths()
        # No-load initialization automatically includes transformer phase shifts.
        self.flat_voltage = self._linear_solve(np.zeros(len(self.keys), complex))
        if np.any(np.abs(self.flat_voltage) < .1*self.nominal_v):
            raise ValueError("Network contains an unenergized phase or an island.")

    def _terminal_model(self, device):
        ph = list(device.get("terminal_nodes", device["phases"]))
        bus = str(device["bus"]).lower()
        ids = np.array([self.index[(bus, p)] for p in ph])
        c = connection_map(ph, str(device.get("connection", "wye")).lower())
        vbase = coil_voltage(device, len(ph), float(self.bus_by_name[bus]["nominal_kv_ll"]))
        return ids, c, vbase

    def _tap_pu(self, name):
        tap = self.regulators[name]["tap"]
        step = (float(tap["max_pu"])-float(tap["min_pu"]))/int(tap["num_taps"])
        value = 1+self.tap_positions[name]*step
        if not float(tap["min_pu"])-1e-12 <= value <= float(tap["max_pu"])+1e-12:
            raise ValueError(f"Out-of-range regulator tap for {name}: {value}")
        return value

    def _assemble(self):
        rows, cols, values = [], [], []
        def stamp(nodes, matrix):
            rr, cc = np.meshgrid(nodes, nodes, indexing="ij")
            rows.extend(rr.ravel()); cols.extend(cc.ravel()); values.extend(matrix.ravel())
        self.elements: list[Element] = []
        for branch in self.network.branches:
            name = str(branch["id"])
            element = make_branch(branch, self.network, self.index, self._tap_pu(name) if name in self.regulators else 1)
            self.elements.append(element)
            stamp(element.nodes, element.y)
        for ids, y in self.shunts:
            stamp(ids, y)
        if self.source_y is not None:
            stamp(self.source_indices, self.source_y)
        self.ybus = coo_matrix((values, (rows, cols)), shape=(len(self.keys), len(self.keys))).tocsc()
        # Equilibrate the 115 kV / 4.16 kV / 0.48 kV voltage levels before LU.
        scale = diags(self.nominal_v[self.free])
        try:
            self.lu = splu((scale @ self.ybus[self.free, :][:, self.free] @ scale).tocsc())
        except RuntimeError as exc:
            raise ValueError("Singular network: check connectivity, open switches and source.") from exc
        self.source_rhs = self.ybus[self.free, :][:, self.fixed] @ self.emf if len(self.fixed) else np.zeros(len(self.free), complex)
        self.factorizations += 1

    def _linear_solve(self, injection):
        rhs = injection[self.free]+self.norton[self.free]-self.source_rhs
        voltage = np.zeros(len(self.keys), complex)
        voltage[self.free] = self.nominal_v[self.free]*self.lu.solve(self.nominal_v[self.free]*rhs)
        if len(self.fixed):
            voltage[self.fixed] = self.emf
        return voltage

    def _device_currents(self, voltage, point, commands):
        injection = np.zeros(len(self.keys), complex)
        loads = {name.lower(): value for name, value in point.load_pq_kva.items()}
        der = {name.lower(): cmd for name, cmd in commands.items()}
        for name, (device, ids, c, vbase, model) in self.device_models.items():
            if not device.get("in_service", True):
                continue
            kind = str(device["kind"])
            if kind == "load":
                power = -complex(loads[name])*1000
            else:
                command = der[name]
                power = complex(command.p_kw, command.q_kvar)*1000
                if kind == "ev":
                    power = -power
                elif kind == "bess" and abs(command.p_kw) < 1e-10:
                    # AC standby demand; SOC is advanced only by ProfileStore.
                    power -= float(device.get("idling_kw_pct", 1))/100*float(device["p_rated_kw"])*1000
            across = c @ voltage[ids]
            if np.any(np.abs(across) < .1*vbase):
                raise RuntimeError(f"Voltage below 0.1 pu at {device['id']}.")
            s, ratio = power/len(c), np.abs(across)/vbase
            low = float(device.get("vmin_pu", .95 if kind in {"load", "ev"} else .9))
            high = float(device.get("vmax_pu", 1.05 if kind in {"load", "ev"} else 1.1))
            if not 0 < low < high:
                raise ValueError(f"Invalid voltage model limits for {device['id']}.")
            if model == 2:
                current = np.conj(s)*across/vbase**2
            else:
                current = np.conj(s/across) if model == 1 else np.conj(s)/vbase*across/np.abs(across)
                # Continuous impedance extension outside the device voltage band.
                for mask, boundary in ((ratio < low, low), (ratio > high, high)):
                    current[mask] = np.conj(s)*across[mask]/(vbase**2*boundary**(2 if model == 1 else 1))
            injection[ids] += c.T @ current
        return injection

    def _solve_network(self, point, commands, initial):
        voltage = initial.copy()
        for iteration in range(1, self.max_iterations+1):
            updated = self._linear_solve(self._device_currents(voltage, point, commands))
            delta = float(np.max(np.abs(updated-voltage)/self.nominal_v))
            voltage = updated
            if not np.all(np.isfinite(voltage)):
                raise RuntimeError(f"Non-finite voltage at {point.time}.")
            if delta < self.voltage_tolerance_pu:
                inj = self._device_currents(voltage, point, commands)
                residual = float(np.max(np.abs((self.ybus @ voltage-self.norton-inj)[self.free])))
                limit = 1e-5+1e-7*max(1., float(np.max(np.abs(inj))))
                if residual > limit:
                    raise RuntimeError(f"FEM KCL residual {residual:g} A exceeds {limit:g} A.")
                return voltage, iteration, residual
        raise RuntimeError(f"FEM current-injection iteration did not converge at {point.time}.")

    def _adjust_taps(self, voltage):
        if self.regulator_mode == "fixed":
            return False
        changed = False
        for element in self.elements:
            branch, name = element.data, str(element.data["id"])
            if name not in self.regulators or not branch.get("in_service", True):
                continue
            control = branch.get("control", {})
            if not control or not control.get("enabled", True):
                continue
            if control.get("type", "voltage_regulator") != "voltage_regulator":
                raise NotImplementedError(f"Unsupported regulator control on {name}.")
            currents = element.y @ voltage[element.nodes]
            secondary, outgoing = voltage[element.nodes[element.nleft]], -currents[element.nleft]
            compensated = secondary/float(control["ptratio"])-complex(control.get("r_v", 0), control.get("x_v", 0))*outgoing/float(control.get("ctprim_a", 1))
            error = float(control["vreg_v"])-abs(compensated)
            if abs(error) <= float(control["band_v"])/2:
                continue
            tap = branch["tap"]
            step = (float(tap["max_pu"])-float(tap["min_pu"]))/int(tap["num_taps"])
            lower, upper = [int(round((float(tap[k])-1)/step)) for k in ("min_pu", "max_pu")]
            position = int(np.clip(self.tap_positions[name]+(1 if error > 0 else -1), lower, upper))
            if position != self.tap_positions[name]:
                self.tap_positions[name] = position
                changed = True
        if changed:
            self._assemble()
        return changed

    def solve(self, operating_point: OperatingPoint, state: PFState | None, context: SolverContext | None = None) -> PFResult:
        if self.network is None:
            raise RuntimeError("Call build(network) before solve().")
        context = context or SolverContext()
        expected_loads = {self.network.device_ids[i].lower() for i in self.network.load_indices}
        expected_der = {self.network.device_ids[i].lower() for i in self.network.der_indices}
        if {k.lower() for k in operating_point.load_pq_kva} != expected_loads or {k.lower() for k in operating_point.der_commands} != expected_der:
            raise ValueError("Operating point must contain exactly every load and DER.")
        if state is not None and "tap_positions" in state.backend_state:
            taps = dict(state.backend_state["tap_positions"])
            if set(taps) != set(self.tap_positions):
                raise ValueError("Warm-start taps do not match this network.")
            if taps != self.tap_positions:
                self.tap_positions = taps
                self._assemble()
        initial = self.flat_voltage.copy()
        if state is not None:
            for i, (bus, ph) in enumerate(self.keys):
                if f"{bus}.{ph}" in state.voltage_guess:
                    initial[i] = state.voltage_guess[f"{bus}.{ph}"]*self.nominal_v[i]
        commands = dict(operating_point.der_commands)
        if any(c.controller_type not in {"constant_pq", "soc_schedule", "volt_var"} for c in commands.values()):
            raise NotImplementedError("Unsupported DER controller.")
        solves = total_iterations = controls = tap_iterations = 0
        while True:
            voltage, iterations, residual = self._solve_network(operating_point, commands, initial)
            solves += 1
            total_iterations += iterations
            initial = voltage
            if self._adjust_taps(voltage):
                tap_iterations += 1
                if tap_iterations > self.max_regulator_iterations:
                    raise RuntimeError(f"Regulator controls did not settle at {operating_point.time}.")
                continue
            changes = {}
            for name, command in commands.items():
                if command.controller_type != "volt_var":
                    continue
                device = self.device_models[name.lower()][0]
                ph = command.parameters.get("terminal_nodes", device["phases"])
                ids = [self.index[(str(device["bus"]).lower(), int(p))] for p in ph]
                measured = float(np.mean(np.abs(voltage[ids])/self.nominal_v[ids]))
                q = command.q_kvar+context.relaxation*(command.adjusted_q(measured)-command.q_kvar)
                changes[name] = command.with_q(q)
            if not changes or max(abs(changes[k].q_kvar-commands[k].q_kvar) for k in changes) <= context.control_tolerance:
                break
            controls += 1
            if controls > context.max_control_iterations:
                raise RuntimeError(f"Volt-VAR controls did not converge at {operating_point.time}.")
            commands.update(changes)
        pu = voltage/self.nominal_v
        voltage_map = {f"{bus}.{ph}": complex(pu[i]) for i, (bus, ph) in enumerate(self.keys)}
        records = tuple({"bus": bus, "node": ph, "v_pu": float(abs(pu[i])), "angle_deg": float(np.angle(pu[i], deg=True))}
                        for i, (bus, ph) in enumerate(self.keys))
        branches, losses, currents_by_name = [], 0j, {}
        for e in self.elements:
            branch, n = e.data, e.nleft
            name, kind = str(branch["id"]), str(branch["kind"])
            local_v = voltage[e.nodes]
            currents = e.y @ local_v
            power = local_v*np.conj(currents)/1000
            losses += np.sum(power)
            currents_by_name[name] = currents
            max_current = float(np.max(np.abs(currents[:n])))
            if kind in {"transformer", "regulator"} and branch["windings"][0]["connection"] == "wye":
                max_current = max(max_current, float(abs(np.sum(currents[:n]))))
            branches.append({"branch": name, "branch_kind": kind, "bus1": str(branch["from_bus"]), "bus2": str(branch["to_bus"]),
                             "max_current_a": max_current,
                             "tap_pu": self._tap_pu(name) if name in self.regulators else (1. if kind == "transformer" else float("nan")),
                             "tap_position": float(self.tap_positions[name]) if name in self.regulators else float("nan"),
                             "terminal1_p_kw": float(np.sum(power[:n]).real), "terminal1_q_kvar": float(np.sum(power[:n]).imag),
                             "terminal1_apparent_kva": float(np.sum(np.abs(power[:n])))})
        # Report line/transformer losses, like the existing QSTS backend.
        # Capacitor reactive injection is not a network loss.
        inj = self._device_currents(voltage, operating_point, commands)
        source_current = (self.source_y @ (self.emf-voltage[self.source_indices]) if self.source_y is not None
                          else (self.ybus @ voltage-inj)[self.source_indices])
        source_power = -np.sum(voltage[self.source_indices]*np.conj(source_current))/1000
        self.last_voltage, self.last_element_currents, self.last_time = voltage, currents_by_name, operating_point.time
        return PFResult(converged=True, bus_voltages=voltage_map, bus_voltage_records=records,
                        branch_records=tuple(branches), iterations=iterations,
                        summary={"converged": True, "min_v_pu": float(np.min(abs(pu))), "max_v_pu": float(np.max(abs(pu))),
                                 "loss_kw": float(losses.real), "loss_kvar": float(losses.imag),
                                 "source_p_kw": float(source_power.real), "source_q_kvar": float(source_power.imag),
                                 "max_kcl_residual_a": residual, "regulator_iterations": tap_iterations},
                        next_state=PFState(voltage_guess=voltage_map, backend_state={"tap_positions": dict(self.tap_positions)}),
                        final_der_commands=commands, control_iterations=controls, pf_solve_count=solves,
                        pf_iterations_total=total_iterations, pf_iterations_last=iterations,
                        metadata={"backend": "native_graph_fem", "path_count": len(self.paths),
                                  "factorizations": self.factorizations, "regulator_mode": self.regulator_mode})

    def _build_paths(self):
        adjacency, edge_elements = defaultdict(list), defaultdict(list)
        for e in self.elements:
            b = e.data
            if not b.get("in_service", True) or (b["kind"] == "switch" and not b.get("normally_closed", True)):
                continue
            edge_elements[frozenset((str(b["from_bus"]).lower(), str(b["to_bus"]).lower()))].append(e)
        for edge in edge_elements:
            if len(edge) != 2:
                raise ValueError("Self-loop branch is unsupported.")
            a, b = sorted(edge)
            adjacency[a].append(b); adjacency[b].append(a)
        slack = str(self.network.base["slack_bus"]).lower()
        parent, order, children = {slack: None}, [slack], defaultdict(list)
        for bus in order:
            for nxt in sorted(adjacency[bus]):
                if nxt == parent[bus]:
                    continue
                if nxt in parent:
                    raise ValueError("Path export requires a radial physical feeder.")
                parent[nxt] = bus
                children[bus].append(nxt)
                order.append(nxt)
        if set(parent) != set(self.bus_by_name):
            raise ValueError("Physical network is disconnected.")
        self.paths = []
        starts = [slack]+[b for b in order if len(children[b]) > 1 and b != slack]
        for start in starts:
            for nxt in children[start]:
                buses, x, edges = [start], [0.], []
                while True:
                    elements = edge_elements[frozenset((buses[-1], nxt))]
                    edges.append([str(e.data["id"]) for e in elements])
                    x.append(x[-1]+max(e.length_km for e in elements))
                    buses.append(nxt)
                    if len(children[nxt]) != 1:
                        break
                    nxt = children[nxt][0]
                self.paths.append({"id": f"path_{len(self.paths)+1}", "buses": buses, "x_km": x, "edges": edges})

    def path_profile_records(self) -> list[dict[str, Any]]:
        if not hasattr(self, "last_voltage"):
            raise RuntimeError("Call solve() before requesting path profiles.")
        by_name, records = {str(e.data["id"]): e for e in self.elements}, []
        for path in self.paths:
            for pos, (bus, distance) in enumerate(zip(path["buses"], path["x_km"])):
                outgoing = path["edges"][pos] if pos < len(path["edges"]) else []
                for ph in self.bus_by_name[bus]["phases"]:
                    idx, current, names = self.index[(bus, ph)], 0j, []
                    for name in outgoing:
                        hits = np.flatnonzero(by_name[name].nodes == idx)
                        if len(hits):
                            current += np.sum(self.last_element_currents[name][hits]); names.append(name)
                    records.append({"timestamp": self.last_time.isoformat(), "path": path["id"], "bus": bus, "x_km": distance, "phase": ph,
                                    "v_pu": float(abs(self.last_voltage[idx])/self.nominal_v[idx]),
                                    "angle_deg": float(np.angle(self.last_voltage[idx], deg=True)),
                                    "outgoing_branch": ";".join(names), "outgoing_current_a": float(abs(current)) if names else None})
        return records
