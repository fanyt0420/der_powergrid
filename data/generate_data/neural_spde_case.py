"""Export a saved neural weather member to a standard QSTS case, optionally run it."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.neural_case import build_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", type=Path, default=ROOT/"output/spde/neural_v1/neural_weather_72h.npz")
    parser.add_argument("--prepared", type=Path, default=ROOT/"data/processed/spde/multiyear_2015_2019")
    parser.add_argument("--layout", type=Path, default=ROOT/"data/config/spde_station_example.json")
    parser.add_argument("--physics", type=Path, default=ROOT/"data/config/spde_local_example.json",
                        help="Only PV/wind physics settings are used; no local random field is added.")
    parser.add_argument("--member", type=int, default=0, help="Zero-based saved weather member.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--hours", type=int, help="Number of saved hourly steps; default: all remaining.")
    parser.add_argument("--case", type=Path, help="Empty case directory; default: data/ieee13_neural_memberNNN.")
    parser.add_argument("--run", action="store_true", help="Run the existing run.py after successful export.")
    parser.add_argument("--output", type=Path, help="QSTS results directory, only with --run.")
    args = parser.parse_args()
    if args.member < 0 or args.start_index < 0 or (args.hours is not None and args.hours < 2):
        parser.error("Use a nonnegative member/start-index and at least two hours.")
    if args.output is not None and not args.run:
        parser.error("--output selects QSTS results and requires --run.")
    case = (args.case or ROOT/"data"/f"ieee13_neural_member{args.member:03d}").resolve()
    metadata = build_case(args.weather, args.prepared, args.layout, args.physics, case,
                          member=args.member, start_index=args.start_index, hours=args.hours)
    print(f"Exported neural weather member {args.member}, {metadata['hours']} hours: {case}", flush=True)
    if args.run:
        command = [sys.executable, str(ROOT/"run.py"), "--data-dir", str(case)]
        if args.output is not None:
            command.extend(["--output", str(args.output.resolve())])
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        print(f'Run: "{sys.executable}" "{ROOT / "run.py"}" --data-dir "{case}"', flush=True)


if __name__ == "__main__":
    main()
