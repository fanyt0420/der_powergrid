"""Secondary synthetic size sweep; correctness and timing, not a speed claim."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fem_radial_experiment import generate_case
from experiments.ieee13_derived_benchmark import compare, environment
from src.power_flow import NetworkModel
from src.profile_store import ProfileStore
from src.qsts import run_qsts
from src.solvers.fem_solver import FiniteElementPFSolver
from src.solvers.opendss_solver import OpenDSSSolver


def run(sizes: list[int], repeats: int, hours: int) -> list[dict]:
    if repeats < 3 or hours < 1:
        raise ValueError("Need at least three repetitions and one hour")
    rows: list[dict] = []
    for n in sizes:
        case_dir = ROOT / "work" / f"fem_scaling_{n}"
        generate_case(case_dir, n, hours)
        network = NetworkModel.from_json(case_dir / "network.json")
        profiles = ProfileStore.from_case(network, case_dir)
        samples: dict[str, list[float]] = defaultdict(list)
        first: dict[str, tuple] = {}
        for rep in range(repeats + 1):
            for label in (("fem", "opendss") if rep % 2 == 0 else ("opendss", "fem")):
                solver = FiniteElementPFSolver() if label == "fem" else OpenDSSSolver(case_dir / "network.dss")
                solver.build(network)
                start = perf_counter()
                result = run_qsts(solver, profiles, network)
                elapsed = perf_counter() - start
                if rep == 0:
                    first[label] = result
                else:
                    samples[label].append(elapsed)
        metric = compare(first["fem"], first["opendss"])
        rows.append({
            "segments": n, "phase_nodes": sum(len(bus["phases"]) for bus in network.buses),
            "loads": len(network.load_indices), "der": len(network.der_indices), "hours": hours,
            "accuracy": metric,
            "fem_qsts_median_s": float(np.median(samples["fem"])),
            "fem_qsts_p25_s": float(np.percentile(samples["fem"], 25)),
            "fem_qsts_p75_s": float(np.percentile(samples["fem"], 75)),
            "opendss_qsts_median_s": float(np.median(samples["opendss"])),
            "opendss_qsts_p25_s": float(np.percentile(samples["opendss"], 25)),
            "opendss_qsts_p75_s": float(np.percentile(samples["opendss"], 75)),
            "fem_samples_s": samples["fem"], "opendss_samples_s": samples["opendss"],
        })
    output = ROOT / "output" / "fem_scaling_study"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({
        "environment": environment(),
        "classification": "synthetic scaling stress test; not an IEEE or field-data benchmark",
        "method": "one warmup, alternating order, same process, build excluded from QSTS time, CSV writes excluded",
        "repetitions": repeats, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in row.items() if k not in {"fem_samples_s", "opendss_samples_s"}} for row in rows], ensure_ascii=False, indent=2))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[30, 60, 120, 240])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()
    run(args.sizes, args.repeats, args.hours)
