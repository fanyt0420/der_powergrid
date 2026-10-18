"""Independent physics checks on phase connections, source and nonlinear loads."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import unittest

import numpy as np

from src.power_flow import DERCommand, NetworkModel, OperatingPoint
from src.solvers.fem_elements import connection_map, source_impedance
from src.solvers.fem_solver import FiniteElementPFSolver

ROOT = Path(__file__).resolve().parents[1]


class FEMPhysicsTest(unittest.TestCase):
    def test_delta_current_conservation(self):
        for phases in ([1, 2, 3], [2, 3]):
            matrix = connection_map(phases, "delta")
            currents = matrix.T @ (np.arange(len(matrix)) + 1j)
            self.assertAlmostEqual(abs(np.sum(currents)), 0)

    def test_source_short_circuit_ratings(self):
        source = {"base_kv_ll": 115, "mvasc3": 20000, "mvasc1": 21000}
        z = source_impedance(source)
        z1, z0 = z[0, 0]-z[0, 1], z[0, 0]+2*z[0, 1]
        self.assertAlmostEqual(115**2/abs(z1), 20000)
        self.assertAlmostEqual(3*115**2/abs(2*z1+z0), 21000)

    def test_transformer_phase_shift_and_voltage_levels_without_load(self):
        network = NetworkModel.from_json(ROOT/"data/ieee13_unbalanced_der/network.json")
        network = replace(network, devices=(), shunt_devices=())
        solver = FiniteElementPFSolver(regulator_mode="fixed")
        solver.build(network)
        result = solver.solve(OperatingPoint(datetime(2026, 1, 1), {}, {}), None)
        self.assertLess(abs(np.angle(result.bus_voltages["650.1"], deg=True)), .02)
        self.assertLess(abs(abs(result.bus_voltages["634.1"])-1.0001), .001)

    def test_bess_charge_discharge_sign_and_standby(self):
        network = NetworkModel.from_json(ROOT/"data/ieee13_unbalanced_der/network.json")
        device = next(d for d in network.devices if d["kind"] == "bess")
        network = replace(network, devices=(device,))
        solver = FiniteElementPFSolver(regulator_mode="fixed")
        solver.build(network)
        powers = []
        for p in (-10, 0, 10):
            command = {device["id"]: DERCommand(p, 0, "constant_pq")}
            point = OperatingPoint(datetime(2026, 1, 1), {}, command)
            current = solver._device_currents(solver.flat_voltage, point, command)
            powers.append(np.sum(solver.flat_voltage*np.conj(current)).real/1000)
        self.assertLess(powers[0], -9)
        self.assertAlmostEqual(powers[1], -float(device["p_rated_kw"])*.01, places=3)
        self.assertGreater(powers[2], 9)


if __name__ == "__main__":
    unittest.main()
