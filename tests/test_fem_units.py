"""Regression checks for physical line length / impedance unit conversion."""

import unittest

import numpy as np

from src.power_flow import NetworkModel
from src.solvers.fem_solver import FiniteElementPFSolver


class FEMLineUnitsTest(unittest.TestCase):
    def test_feet_against_ohms_per_mile(self) -> None:
        network = NetworkModel(
            base={"frequency_hz": 60, "base_kv_ll": 4.16, "slack_bus": "s"},
            source={"bus": "s", "phases": [1, 2, 3], "base_kv_ll": 4.16, "voltage_pu": 1.0},
            buses=(
                {"id": "s", "phases": [1, 2, 3], "nominal_kv_ll": 4.16, "is_slack": True},
                {"id": "b", "phases": [1, 2, 3], "nominal_kv_ll": 4.16},
            ),
            branches=({
                "id": "line", "kind": "line", "from_bus": "s", "to_bus": "b",
                "phases": [1, 2, 3], "length": 5280, "length_unit": "ft", "line_code": "code",
            },),
            devices=(),
            line_codes={"code": {
                "unit": "mi", "phases": [1, 2, 3],
                "r_ohm_per_unit": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "x_ohm_per_unit": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            }},
        )
        solver = FiniteElementPFSolver()
        solver.build(network)
        np.testing.assert_allclose(solver.elements[0][3], np.eye(3) * (1 + 2j))


if __name__ == "__main__":
    unittest.main()
