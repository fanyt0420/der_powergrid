"""Behavior checks for the coupled model, not snapshots of implementation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.power_flow import NetworkModel
from src.dynamics.model import KuramotoDAE
from src.dynamics.network import PassiveNetwork
from src.dynamics.run import ROOT, run_case, validate_config
from src.dynamics.snapshot import capture_snapshot, check_network
from src.dynamics.solver import algebraic_solve, equilibrium, modes, step


class CoupledDAETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ROOT / "data/ieee13_unbalanced_der"
        cls.config = json.loads((cls.data / "dynamic_config.json").read_text(encoding="utf-8"))
        cls.network = NetworkModel.from_json(cls.data / "network.json")
        cls.snapshot = capture_snapshot(cls.network, cls.data)

    def model(self):
        p = PassiveNetwork(self.network, self.snapshot["nodes"], self.snapshot["taps"], self.config)
        return KuramotoDAE(self.network, p, self.snapshot, copy.deepcopy(self.config))

    def test_network_and_original_pq_regression(self):
        m = self.model()
        self.assertTrue(check_network(m.passive, self.snapshot)["passed"])
        x, y = m.initial_guess()
        y, _ = algebraic_solve(m, x, y, all_pq=True)
        error = max(abs(m.voltage(y) - self.snapshot["voltage"]) / m.passive.vbase)
        self.assertLess(error, 1e-6)
        self.assertEqual((m.nx, m.ny), (17, 76))

    def test_equilibrium_and_no_disturbance(self):
        m = self.model()
        x, y, info = equilibrium(m)
        self.assertEqual(info["status"], "converged")
        self.assertEqual(info["equations"], 78)
        self.assertLess(max(abs(m.evaluate(x, y)[0])), 1e-5)
        xn, yn, _ = step(m, x, y, .005)
        self.assertLess(max(abs(xn - x)), 1e-6)
        self.assertLess(max(abs(m.kcl(xn, yn))), self.config["tolerance"])
        # Stability is a calculated result, not a hard-coded property of a configuration.
        self.assertEqual(len(modes(m, x, y)["modes"]), m.nx)

    def test_trip_removes_current_and_freezes_states(self):
        m = self.model()
        x, y, _ = equilibrium(m)
        tripped = {"PV_671_A"}
        yp, _ = algebraic_solve(m, x, y, tripped=tripped)
        dx, _, records, _, _ = m.evaluate(x, yp, tripped=tripped)
        pv = next(r for r in records if r["device_id"] == "PV_671_A")
        self.assertEqual(pv["current_max_a"], 0)
        self.assertTrue(np.isnan(pv["delta_f_hz"]))
        device = next(d for d in m.dynamic if d["device_id"] == "PV_671_A")
        self.assertTrue(np.all(dx[device["slice"]] == 0))
        self.assertLess(max(abs(m.kcl(x, yp, tripped=tripped))), self.config["tolerance"])
        load_before = m.evaluate(x, y)[3]
        load_after = m.evaluate(x, y, tripped={"BESS_671_B"})[3]
        self.assertAlmostEqual(float((load_before - load_after).real), 1000, places=5)

    def test_off_grid_event_and_failure_artifact(self):
        config = copy.deepcopy(self.config)
        config["t_end_s"] = .025
        config["scenarios"] = {"pv_trip": [{"time_s": .012, "type": "der_trip", "device_id": "PV_671_A"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = run_case(self.data, path, config, "pv_trip", snapshot=self.snapshot)
            self.assertTrue(result["success"])
            frame = pd.read_csv(path / "dae_state.csv")
            pre = frame[frame.event_side == "pre_event"].iloc[0]
            post = frame[frame.event_side == "post_event"].iloc[0]
            self.assertAlmostEqual(post.time_s, .012)
            np.testing.assert_allclose(pre.iloc[2:].to_numpy(float), post.iloc[2:].to_numpy(float), atol=0, rtol=0)
            config["dynamic_devices"][2]["current_limit_pu"] = .01
            result = run_case(self.data, path, config, "pv_trip", snapshot=self.snapshot)
            self.assertFalse(result["success"])
            failure = json.loads((path / "dae_failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["failure_stage"], "out_of_model_domain")
            self.assertEqual(failure["device_violations"][0]["device_id"], "Wind_680_ABC")

    def test_invalid_step_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["dt_s"] = 0
        with self.assertRaises(ValueError):
            validate_config(config, self.network)

    def test_invalid_config_preserves_existing_results(self):
        config = copy.deepcopy(self.config)
        config["dt_s"] = 0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            existing = output / "dae_initialization.json"
            existing.write_text('{"previous_result": true}')
            with self.assertRaises(ValueError):
                run_case(self.data, output, config)
            self.assertEqual(existing.read_text(encoding="utf-8"), '{"previous_result": true}')
            config = copy.deepcopy(self.config)
            config["source_boundary"] = "unknown"
            with self.assertRaisesRegex(ValueError, "source_boundary"):
                run_case(self.data, output, config)
            self.assertEqual(existing.read_text(encoding="utf-8"), '{"previous_result": true}')
        for iterations in (True, 1.5, float("nan"), 0):
            with self.subTest(iterations=iterations):
                config = copy.deepcopy(self.config)
                config["max_newton_iterations"] = iterations
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    validate_config(config, self.network)

    def test_initial_event_has_both_sides(self):
        config = copy.deepcopy(self.config)
        config["t_end_s"] = .005
        config["scenarios"] = {"pv_trip": [{"time_s": 0, "type": "der_trip", "device_id": "PV_671_A"}]}
        with tempfile.TemporaryDirectory() as directory:
            result = run_case(self.data, directory, config, "pv_trip", snapshot=self.snapshot)
            self.assertTrue(result["success"])
            states = pd.read_csv(Path(directory) / "dae_state.csv")
            self.assertEqual(list(states.loc[states.time_s == 0, "event_side"]), ["pre_event", "post_event"])
            np.testing.assert_allclose(states.iloc[0, 2:].to_numpy(float), states.iloc[1, 2:].to_numpy(float), atol=0, rtol=0)

    def test_different_topology_and_instance_names(self):
        devices = []
        config = copy.deepcopy(self.config)
        config["dynamic_devices"] = config["dynamic_devices"][:2]
        powers = {}
        for i, spec in enumerate(config["dynamic_devices"]):
            device = dict(next(d for d in self.network.devices if d["id"] == spec["device_id"]))
            device.update(id=f"unit_{i}", bus="leaf")
            spec["device_id"] = device["id"]
            devices.append(device)
            powers[device["id"]] = complex(10000, 500)
        network = NetworkModel(
            base={"frequency_hz":60, "base_kv_ll":4.16, "slack_bus":"grid"},
            source={"bus":"grid", "phases":[1,2,3], "base_kv_ll":4.16, "voltage_pu":1},
            buses=({"id":"grid", "phases":[1,2,3], "nominal_kv_ll":4.16, "is_slack":True},
                   {"id":"leaf", "phases":[1,2,3], "nominal_kv_ll":4.16}),
            branches=({"id":"feeder", "kind":"line", "from_bus":"grid", "to_bus":"leaf",
                       "phases":[1,2,3], "length":1, "length_unit":"km", "line_code":"c"},),
            devices=tuple(devices),
            line_codes={"c":{"phases":[1,2,3], "unit":"km",
                             "r_ohm_per_unit":(np.eye(3)*.2).tolist(),
                             "x_ohm_per_unit":(np.eye(3)*.3).tolist()}})
        # A different node count and deliberately permuted phase order.
        nodes = ("leaf.3", "grid.2", "leaf.1", "grid.1", "leaf.2", "grid.3")
        angles = {1:0, 2:-2*np.pi/3, 3:2*np.pi/3}
        snapshot = {"nodes":nodes, "taps":{}, "loads_va":{}, "der_va":powers,
                    "voltage":np.array([4160/np.sqrt(3)*np.exp(1j*angles[int(n[-1])]) for n in nodes])}
        passive = PassiveNetwork(network, nodes, {}, config)
        model = KuramotoDAE(network, passive, snapshot, config)
        x, y, info = equilibrium(model)
        self.assertEqual((model.nx, model.ny, info["equations"]), (11,6,8))
        self.assertEqual(info["status"], "converged")
        xn, yn, _ = step(model, x, y, .005)
        self.assertLess(max(abs(model.kcl(xn,yn))), config["tolerance"])
        config["source_boundary"] = "nominal_ideal"
        ideal = KuramotoDAE(network, passive, snapshot, config)
        np.testing.assert_allclose(ideal.fixed_voltage, snapshot["voltage"][passive.fixed])

    def test_terminal_order_controls_phase_rotation(self):
        from dataclasses import replace
        original = self.model()
        x, y, _ = equilibrium(original)
        devices = [dict(d) for d in self.network.devices]
        next(d for d in devices if d["id"] == "Wind_680_ABC")["terminal_nodes"] = [3,1,2]
        network = replace(self.network, devices=tuple(devices))
        model = KuramotoDAE(network, original.passive, self.snapshot, copy.deepcopy(self.config))
        before = original.evaluate(x,y)
        after = model.evaluate(x,y)
        np.testing.assert_allclose(after[0],before[0],atol=1e-10)
        np.testing.assert_allclose(after[1],before[1],atol=1e-9)

    def test_gfl_pll_and_instance_expansion(self):
        m = self.model()
        x, y, _ = equilibrium(m)
        pv = next(d for d in m.dynamic if d["device_id"] == "PV_671_A")
        disturbed = x.copy()
        disturbed[pv["slice"].start] += .01
        self.assertLess(m.evaluate(disturbed, y)[0][pv["slice"].start], 0)
        config = copy.deepcopy(self.config)
        config["dynamic_devices"].reverse()
        other = KuramotoDAE(self.network, m.passive, self.snapshot, config)
        _, yo, _ = equilibrium(other)
        np.testing.assert_allclose(other.voltage(yo), m.voltage(y), atol=.001, rtol=0)
        # Only configuration fields belong in a new instance.
        extra = copy.deepcopy(next(d for d in self.config["dynamic_devices"] if d["device_id"] == "PV_671_A"))
        extra["device_id"] = "PV_670_C"
        config["dynamic_devices"].append(extra)
        larger = KuramotoDAE(self.network, m.passive, self.snapshot, config)
        xl, yl, info = equilibrium(larger)
        self.assertEqual((larger.nx, larger.ny, info["equations"]), (23, 76, 78))
        self.assertLess(max(abs(larger.evaluate(xl, yl)[0])), 1e-5)

    def test_invalid_device_instances_are_rejected(self):
        cases = [
            ("model", "typo"),
            ("tau_p_s", 0), ("r_c_pu", -0.1),
            ("D", float("nan")), ("s_ref_va", True),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.config)
                config["dynamic_devices"][0][key] = value
                with self.assertRaises(ValueError):
                    validate_config(config, self.network)
                # Direct model construction must enforce the same contract.
                with self.assertRaises(ValueError):
                    KuramotoDAE(self.network, None, None, config)
        config = copy.deepcopy(self.config)
        config["dynamic_devices"].append(copy.deepcopy(config["dynamic_devices"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_config(config, self.network)
        from types import SimpleNamespace
        network = SimpleNamespace(devices=[dict(d) for d in self.network.devices])
        next(d for d in network.devices if d["id"] == config["dynamic_devices"][0]["device_id"])["in_service"] = False
        with self.assertRaisesRegex(ValueError, "out of service"):
            validate_config(self.config, network)


if __name__ == "__main__":
    unittest.main()
