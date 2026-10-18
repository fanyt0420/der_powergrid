"""Acceptance checks on both complete, unmodified repository DER datasets."""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from test.compare_solvers import CASES, ROOT, compare_case


class CompleteDERComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = {case: compare_case(case) for case in CASES}

    def test_complete_timelines_and_electrical_residuals(self):
        for case, report in self.reports.items():
            with self.subTest(case=case):
                self.assertEqual(report["time_steps"], 96 if case == "ieee13_measured" else 24)
                self.assertEqual(report["phase_nodes"], 41)
                self.assertEqual(report["branches"], 17)
                self.assertEqual(report["der_devices"], 13)
                self.assertLess(report["fem_max_kcl_residual_a"], 1e-5)
                self.assertEqual(report["auto_control"]["max_soc_difference_pct"], 0)

    def test_same_controls_isolate_network_solution_accuracy(self):
        # Dataset-specific engineering regression gates, not universal FEM bounds.
        for case, report in self.reports.items():
            with self.subTest(case=case):
                matched = report["matched_controls"]
                self.assertLess(matched["max_complex_error_pu"], 3e-4)
                self.assertLess(matched["max_branch_current_difference_a"], .6)
                self.assertLess(matched["loss_difference_kw"], .2)
                auto = report["auto_control"]
                self.assertLess(auto["max_magnitude_error_pu"], .02)
                self.assertLessEqual(auto["max_tap_difference_steps"], 2.000001)


class UnifiedCLITest(unittest.TestCase):
    def test_both_cases_run_without_importing_dss(self):
        # Import guard proves run.py --solver fem does not silently fall back to
        # OpenDSS for primitive matrices, boundary values, controls or solutions.
        script = '''
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "dss" or name.startswith("dss."):
        raise AssertionError("FEM must not import OpenDSS")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ["run.py", "--solver", "fem", "--data-dir", sys.argv[1], "--output", sys.argv[2]]
runpy.run_path("run.py", run_name="__main__")
'''
        with TemporaryDirectory() as tmp:
            for case in CASES:
                with self.subTest(case=case):
                    output = Path(tmp)/case
                    result = subprocess.run([sys.executable, "-c", script, str(ROOT/"data"/case), str(output)],
                                            cwd=ROOT, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = {"qsts_bus_voltages.csv", "qsts_system.csv", "qsts_branch_results.csv",
                                "qsts_constraints.csv", "qsts_hosting_capacity_metrics.csv", "fem_path_profile_last_step.csv"}
                    self.assertTrue(expected.issubset(p.name for p in output.iterdir()))
                    system = pd.read_csv(output/"qsts_system.csv")
                    self.assertTrue(system.converged.all())
                    self.assertEqual(len(system), 96 if case == "ieee13_measured" else 24)
                    profile = pd.read_csv(output/"fem_path_profile_last_step.csv")
                    # Repeated junction nodes on different paths share a voltage.
                    self.assertLess(profile.groupby(["bus", "phase"]).v_pu.agg(lambda v: v.max()-v.min()).max(), 1e-12)
                    branches = pd.read_csv(output/"qsts_branch_results.csv")
                    taps = branches.loc[branches.branch_kind == "regulator", "tap_pu"]
                    self.assertTrue(taps.between(.9, 1.1).all())
                    self.assertGreater(taps.nunique(), 1)
                    for col in system:
                        if col.endswith("_soc_pct"):
                            self.assertTrue(system[col].between(20-1e-9, 100+1e-9).all())
                    self.assertTrue(np.isfinite(branches.max_current_a).all())


if __name__ == "__main__":
    unittest.main()
