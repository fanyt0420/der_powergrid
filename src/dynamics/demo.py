"""Compare two declared GFM reactances under an identical loaded QSTS snapshot.

Run: python -m src.dynamics.demo
This is a research-parameter comparison, not a calibrated equipment claim.
"""

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from src.power_flow import NetworkModel
from src.dynamics.run import ROOT, run_case, write_json
from src.dynamics.snapshot import capture_snapshot
from src.dynamics.plotting import plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ieee13_unbalanced_der")
    parser.add_argument("--timestamp", default="2026-01-01 12:00:00")
    parser.add_argument("--output", type=Path, default=ROOT / "output/dynamic_demo")
    parser.add_argument("--plot-only", action="store_true", help="Redraw recorded runs without simulating again")
    args = parser.parse_args()
    config = json.loads((args.data_dir / "dynamic_config.json").read_text(encoding="utf-8"))
    network = NetworkModel.from_json(args.data_dir / "network.json")
    event = config["scenarios"]["load_step"][0]
    gfm = next(d for d in config["dynamic_devices"] if d["model"] == "inertia")
    devices = {d["id"]: d for d in network.devices}
    gfl = next(d for d in config["dynamic_devices"] if d["model"] == "gfl"
               and len(devices[d["device_id"]]["phases"]) == 1)
    if args.plot_only:
        summary = json.loads((args.output / "comparison.json").read_text(encoding="utf-8"))
        plot(summary, args.output, summary["varied_device"], devices[gfl["device_id"]], network)
        return
    snapshot = capture_snapshot(network, args.data_dir, args.timestamp)
    summary = {"purpose": "research-parameter comparison; no equipment calibration or global stability proof",
               "timestamp": snapshot["timestamp"], "event": event,
               "varied_device": gfm["device_id"], "runs": []}
    for reactance in (.073, .20):
        for dt in (.0025, .00125):
            cfg = copy.deepcopy(config)
            next(d for d in cfg["dynamic_devices"] if d["device_id"] == gfm["device_id"])["x_c_pu"] = reactance
            cfg["dt_s"] = dt
            output = args.output / f"xc_{reactance:g}_dt_{dt:g}"
            result = run_case(args.data_dir, output, cfg, "load_step", snapshot=snapshot)
            if not result["success"]:
                raise RuntimeError(f"Demo failed: {output}: {result['failure']}")
            modes = json.loads((output / "dae_modes.json").read_text(encoding="utf-8"))
            der = pd.read_csv(output / "dae_der.csv")
            row = der[der.device_id == gfm["device_id"]]
            summary["runs"].append({"x_c_pu": reactance, "dt_s": dt,
                "output": str(output.resolve()), "spectral_abscissa_per_s": modes["spectral_abscissa_per_s"],
                "bess_frequency_peak_hz": float(row.delta_f_hz.abs().max()),
                "max_current_ratio": float(der.current_ratio.max())})
            print(f"xc={reactance:g}, dt={dt:g}, alpha={modes['spectral_abscissa_per_s']:.4f}", flush=True)
    summary["peak_step_differences"] = []
    for reactance in (.073, .20):
        coarse, fine = [r for r in summary["runs"] if r["x_c_pu"] == reactance]
        summary["peak_step_differences"].append({"x_c_pu": reactance,
            "relative_difference": abs(coarse["bess_frequency_peak_hz"]-fine["bess_frequency_peak_hz"])
                                   / max(fine["bess_frequency_peak_hz"], 1e-12)})
    write_json(args.output / "comparison.json", summary)
    plot(summary, args.output, gfm["device_id"], devices[gfl["device_id"]], network)



if __name__ == "__main__":
    main()
