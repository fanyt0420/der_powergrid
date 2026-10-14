"""Run the published IEEE 13-node feeder with a fixed regulator configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dss import dss

from src.solvers.dss_graph_fem import DssGraphFEMSolver


ROOT = Path(__file__).resolve().parent
TAPS = {"Reg1": 1.0625, "Reg2": 1.05, "Reg3": 1.06875}


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the IEEE 13-node feeder with graph FEM")
    parser.add_argument("--master", type=Path, default=ROOT / "data" / "ieee13" / "IEEE13Nodeckt.dss")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "ieee13_fem")
    args = parser.parse_args()
    solver = DssGraphFEMSolver(args.master, boundary_bus="650", fixed_taps=TAPS)
    solver.build()
    result = solver.solve()
    reference = solver.reference_voltages()
    rows = []
    for key, value in result["voltages"].items():
        bus, phase = key.rsplit(".", 1)
        dss.ActiveCircuit.SetActiveBus(bus)
        vbase = float(dss.ActiveCircuit.ActiveBus.kVBase) * 1000
        ref = reference[key]
        rows.append({"bus": bus, "phase": int(phase), "fem_v_pu": abs(value) / vbase,
                     "opendss_v_pu": abs(ref) / vbase,
                     "magnitude_error_pu": abs(abs(value) - abs(ref)) / vbase,
                     "complex_error_v": abs(value - ref)})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "ieee13_voltage_comparison.csv", index=False)
    pd.DataFrame(solver.path_profile_records()).to_csv(output / "ieee13_path_profile.csv", index=False)
    current_rows = []
    circuit = dss.ActiveCircuit
    for name in circuit.Lines.AllNames:
        circuit.SetActiveElement(f"Line.{name}")
        element = circuit.ActiveCktElement
        bus = element.BusNames[0].split(".")[0].lower()
        reference_current = list(element.Currents)
        for k, phase in enumerate(list(element.NodeOrder)[:element.NumConductors]):
            if int(phase) == 0:
                continue
            reference_value = complex(reference_current[2 * k], reference_current[2 * k + 1])
            fem_value = solver.element_terminal_currents[f"line.{name.lower()}"].get((bus, int(phase)), 0j)
            current_rows.append({"line": name, "phase": int(phase), "fem_current_a": abs(fem_value),
                                 "opendss_current_a": abs(reference_value),
                                 "complex_error_a": abs(fem_value - reference_value)})
    pd.DataFrame(current_rows).to_csv(output / "ieee13_line_current_comparison.csv", index=False)
    summary = {"case": "IEEE 13-node", "boundary_bus": "650", "fixed_taps": TAPS,
               "nodes_compared": len(rows), "paths": len(solver.paths),
               "iterations": result["iterations"], "max_kcl_residual_a": result["max_residual_a"],
               "max_magnitude_error_pu": max(row["magnitude_error_pu"] for row in rows),
               "max_complex_error_v": max(row["complex_error_v"] for row in rows),
               "max_line_current_complex_error_a": max(row["complex_error_a"] for row in current_rows)}
    (output / "ieee13_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
