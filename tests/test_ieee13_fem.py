import unittest
from pathlib import Path
from dss import dss

from src.solvers.dss_graph_fem import DssGraphFEMSolver


class IEEE13GraphFEMTest(unittest.TestCase):
    def test_fixed_tap_ieee13_matches_reference_voltage(self):
        master = Path(__file__).resolve().parents[1] / "data" / "ieee13" / "IEEE13Nodeckt.dss"
        solver = DssGraphFEMSolver(master, fixed_taps={"Reg1": 1.0625, "Reg2": 1.05, "Reg3": 1.06875})
        solver.build()
        result = solver.solve()
        reference = solver.reference_voltages()
        self.assertEqual(len(result["voltages"]), 38)
        self.assertEqual(len(solver.paths), 9)
        self.assertLess(result["max_residual_a"], 1e-4)
        self.assertLess(max(abs(abs(value) - abs(reference[node])) / abs(reference[node])
                            for node, value in result["voltages"].items()), 2e-4)
        circuit = dss.ActiveCircuit
        circuit.SetActiveElement("Line.650632")
        element = circuit.ActiveCktElement
        raw = list(element.Currents)
        ref = complex(raw[0], raw[1])
        got = solver.element_terminal_currents["line.650632"][("rg60", 1)]
        self.assertLess(abs(got - ref), 1.0)


if __name__ == "__main__":
    unittest.main()
