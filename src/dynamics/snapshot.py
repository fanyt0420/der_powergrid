"""Replay QSTS through the requested instant; export a single coherent snapshot."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dss import dss

from src.power_flow import OperatingPoint, SolverContext
from src.profile_store import ProfileStore
from src.solvers.opendss_solver import OpenDSSSolver


def capture_snapshot(network, data_dir, timestamp=None):
    profiles = ProfileStore.from_case(network, data_dir)
    target = pd.Timestamp(timestamp) if timestamp else profiles.timestamps[0]
    if target not in profiles.timestamps:
        raise ValueError(f"Timestamp {target} is not on the QSTS timeline")
    solver = OpenDSSSolver(data_dir / "network.dss")
    solver.build(network)
    dss("Set Tolerance=1e-11 Maxiterations=100")
    state = None
    hours = network.step_seconds() / 3600
    for k, t in enumerate(profiles.timestamps):
        loads = profiles.load_mapping_at(k)
        commands = profiles.commands_at(k, hours)
        result = solver.solve(OperatingPoint(t.to_pydatetime(), loads, commands), state, SolverContext(time_step_hours=hours))
        state = result.next_state
        if t == target:
            break
        profiles.advance_bess(result.final_der_commands, hours)
    # No Solve after this point: taps, commands and voltages are from this solve.
    dss("Set ControlMode=Off")
    circuit = dss.ActiveCircuit
    nodes = tuple(n.lower() for n in circuit.YNodeOrder)
    raw = np.asarray(circuit.YNodeVarray)
    voltage = raw[::2] + 1j * raw[1::2]
    taps = {r["branch"].lower(): float(r["tap_pu"]) for r in result.branch_records if r["branch_kind"] in {"regulator", "transformer"}}
    references = []
    for category, names in (("Line", circuit.Lines.AllNames), ("Transformer", circuit.Transformers.AllNames), ("Capacitor", circuit.Capacitors.AllNames)):
        for name in names:
            circuit.SetActiveElement(f"{category}.{name}")
            element = circuit.ActiveCktElement
            node_order = np.asarray(element.NodeOrder).reshape(element.NumTerminals, element.NumConductors)
            local_nodes = [f"{bus.split('.')[0].lower()}.{int(p)}" for bus, phases in zip(element.BusNames, node_order) for p in phases]
            keep = [i for i, n in enumerate(local_nodes) if not n.endswith(".0")]
            raw = np.asarray(element.Yprim)
            y = (raw[::2] + 1j * raw[1::2]).reshape(len(local_nodes), len(local_nodes))
            references.append({"name": f"{category}.{name}", "nodes": [local_nodes[i] for i in keep], "y": y[np.ix_(keep, keep)]})
    circuit.SetActiveElement("Vsource.source")
    source_yprim = np.asarray(circuit.ActiveCktElement.Yprim).tolist()
    auxiliary = {}
    for device in network.devices:
        if device["kind"] == "bess":
            circuit.SetActiveElement("Storage." + device["id"])
            raw = circuit.ActiveCktElement.TotalPowers
            consumed = complex(raw[0], raw[1]) * 1000
            command = result.final_der_commands[device["id"]]
            # OpenDSS Storage has idle/conversion losses outside commanded PQ.
            auxiliary[device["id"]] = consumed + complex(command.p_kw, command.q_kvar) * 1000
    circuit.Solution.BuildYMatrix(1, 1)
    raw = np.asarray(circuit.SystemY)
    system_y_series = (raw[::2] + 1j * raw[1::2]).reshape(len(nodes), len(nodes))
    return {
        "timestamp": str(target), "nodes": nodes, "voltage": voltage, "taps": taps,
        "loads_va": {n: s * 1000 for n, s in loads.items()},
        "der_va": {n: complex(c.p_kw, c.q_kvar) * 1000 for n, c in result.final_der_commands.items()},
        "soc_before_window_pct": profiles.state_summary(),
        "references": references, "source_yprim_interleaved_s": source_yprim,
        "auxiliary_consumption_va": auxiliary,
        "system_y_series_siemens": system_y_series,
    }


def check_network(passive, snapshot):
    records = []
    refs = {r["name"].lower(): r for r in snapshot["references"]}
    assembled = np.zeros_like(passive.y)
    for element in passive.elements:
        ref = refs[element.name.lower()]
        names = [passive.nodes[i] for i in element.indices]
        order = [ref["nodes"].index(n) for n in names]
        y = ref["y"][np.ix_(order, order)]
        error = np.abs(element.y - y)
        relative = error / np.maximum(np.abs(y), 1e-3)
        passed = bool(np.all((relative <= 1e-6) | (error <= 1e-8)))
        records.append({"element": element.name, "max_relative_error": float(relative.max()), "max_absolute_error_s": float(error.max()), "passed": passed})
        assembled[np.ix_(element.indices, element.indices)] += y
    absolute = np.abs(passive.y - assembled)
    relative = absolute / np.maximum(np.abs(assembled), 1e-3)
    return {"passed": all(r["passed"] for r in records) and bool(np.all((relative <= 1e-3) | (absolute <= 1e-8))),
            "elements": records, "node_count": len(passive.nodes),
            "assembly_max_relative_error": float(relative.max()),
            "reference_scope": "sum of passive per-element Yprim, including capacitors; no source/load/DER admittances"}


def save_snapshot(snapshot, output):
    import json
    metadata = {k: v for k, v in snapshot.items() if k not in {"voltage", "references", "loads_va", "der_va", "auxiliary_consumption_va", "system_y_series_siemens"}}
    metadata["loads_va"] = {n: [s.real, s.imag] for n, s in snapshot["loads_va"].items()}
    metadata["der_va"] = {n: [s.real, s.imag] for n, s in snapshot["der_va"].items()}
    metadata["auxiliary_consumption_va"] = {n: [s.real, s.imag] for n, s in snapshot["auxiliary_consumption_va"].items()}
    (output / "snapshot.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    arrays = {"voltage_v": snapshot["voltage"], "nodes": np.array(snapshot["nodes"])}
    arrays["system_y_series_siemens"] = snapshot["system_y_series_siemens"]
    passive_y = np.zeros((len(snapshot["nodes"]), len(snapshot["nodes"])), complex)
    for k, ref in enumerate(snapshot["references"]):
        arrays[f"element_{k}_name"] = np.array(ref["name"])
        arrays[f"element_{k}_nodes"] = np.array(ref["nodes"])
        arrays[f"element_{k}_y_siemens"] = ref["y"]
        indices = [snapshot["nodes"].index(node) for node in ref["nodes"]]
        passive_y[np.ix_(indices, indices)] += ref["y"]
    arrays["passive_y_siemens"] = passive_y
    np.savez(output / "network_reference.npz", **arrays)
