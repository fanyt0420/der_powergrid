"""Small-network regression for the ProfileStore/QSTS interface."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.power_flow import NetworkModel
from src.profile_store import ProfileStore
from src.qsts import run_qsts
from src.solvers.fem_solver import FiniteElementPFSolver


class FEMLatestQSTSTest(unittest.TestCase):
    def test_one_step_load_and_path_profile(self) -> None:
        network = NetworkModel(
            base={"frequency_hz": 60, "base_kv_ll": 4.16, "slack_bus": "source"},
            source={"bus": "source", "phases": [1, 2, 3], "base_kv_ll": 4.16, "voltage_pu": 1.0},
            buses=(
                {"id": "source", "phases": [1, 2, 3], "nominal_kv_ll": 4.16, "is_slack": True},
                {"id": "loadbus", "phases": [1, 2, 3], "nominal_kv_ll": 4.16},
            ),
            branches=({"id": "L1", "kind": "line", "from_bus": "source", "to_bus": "loadbus",
                       "phases": [1, 2, 3], "length_km": 1.0,
                       "z_ohm_per_km": [[(0.3, 0.2), (0, 0), (0, 0)],
                                        [(0, 0), (0.3, 0.2), (0, 0)],
                                        [(0, 0), (0, 0), (0.3, 0.2)]]},),
            devices=({"id": "Load1", "kind": "load", "bus": "loadbus", "phases": [1, 2, 3],
                      "connection": "wye", "kv_ll": 4.16},),
            simulation={"start_datetime": "2026-01-01 00:00:00",
                        "end_datetime": "2026-01-01 01:00:00", "step_seconds": 3600},
        )
        with TemporaryDirectory() as folder:
            Path(folder, "load_profiles.csv").write_text(
                "timestamp,load_name,p_kw,q_kvar\n2026-01-01 00:00:00,Load1,300,90\n", encoding="utf-8")
            profiles = ProfileStore.from_case(network, folder)
            solver = FiniteElementPFSolver()
            solver.build(network)
            bus, system, branches = run_qsts(solver, profiles, network)
        self.assertEqual(len(system), 1)
        self.assertEqual(len(bus), 6)
        self.assertEqual(len(branches), 1)
        self.assertLess(float(system.loc[0, "min_v_pu"]), 1.0)
        self.assertEqual(len(solver.paths), 1)
        self.assertEqual(len(solver.path_profile_records()), 6)


if __name__ == "__main__":
    unittest.main()
