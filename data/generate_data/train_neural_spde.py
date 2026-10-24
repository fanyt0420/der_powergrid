"""Train neural and constant-coefficient SPDEs on disjoint ERA5 years.

Models share the same implicit PDE, preprocessing, masked-state-proxy likelihood
and multi-step training loss. Validation selects the checkpoints; a saved July
weather window is exported. All comparisons now use benchmark_spde.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.neural_weather_spde import MultiYearWeather, CoefficientSPDE, rollout_loss


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def monthly_origins(data, split, hours):
    origins = []
    for year in data.manifest["split"][split]:
        for month in range(1, 13):
            first = pd.Timestamp(year, month, 15, 0, 30, tz="UTC")
            index = data.times.get_indexer([first])[0]
            if index >= 24 and index+hours <= len(data.times):
                origins.append(int(index-1))
    return origins


def physical_samples(data, paths, indices):
    # paths: member,time,field,node. Data offsets are fixed by training only.
    raw = paths*data.scale[None, None, :, None]+data.background[indices][None]
    solar = 1.6/(1+np.exp(-np.clip(raw[:, :, 0], -40, 40)))
    return {"ghi": solar*data.clear[indices][None],
            "wind_speed_100m": np.hypot(raw[:, :, 1], raw[:, :, 2]), "temp_air": raw[:, :, 3]}


def physical_truth(data, indices):
    return {"ghi": data.ghi[indices], "wind_speed_100m": np.hypot(data.raw[indices, 1], data.raw[indices, 2]),
            "temp_air": data.raw[indices, 3]}


def crps(truth, ensemble):
    count = len(ensemble)
    ordered = np.sort(ensemble, axis=0)
    coefficients = (2*np.arange(1, count+1)-count-1).reshape((count,)+(1,)*(ensemble.ndim-1))
    return abs(ensemble-truth).mean(axis=0)-(ordered*coefficients).sum(axis=0)/(count*(count-1))


def physical_scales(data):
    indices = np.flatnonzero(data.times.year.isin(data.manifest["split"]["train"]))
    truth = physical_truth(data, indices)
    result = {}
    for name, values in truth.items():
        values = values[:, data.train_sites]
        mask = np.isfinite(values)
        if name == "ghi":
            mask &= data.clear[indices][:, data.train_sites] >= 50
        result[name] = float(values[mask].std())
    return result


@torch.no_grad()
def validate(model, data, device, scales, *, seed, members=16, hours=24):
    scores = []
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for origin in monthly_origins(data, "validation", hours):
            batch = data.batch([origin], model.history, hours, device)
            paths = model.rollout(batch["history"], batch["calendar"], batch["daylight"], members=members)[0].cpu().numpy()
            indices = batch["indices"][0]
            sample, truth = physical_samples(data, paths, indices), physical_truth(data, indices)
            parts = []
            for name in scales:
                target = truth[name][:, data.train_sites]
                predicted = sample[name][:, :, data.train_sites]
                mask = np.isfinite(target)
                if name == "ghi":
                    mask &= data.clear[indices][:, data.train_sites] >= 50
                parts.append(float(crps(target, predicted)[mask].mean()/scales[name]))
            scores.append(float(np.mean(parts)))
    return float(np.mean(scores))


def train_one(role, data, args, device, scales, out):
    torch.manual_seed(args.seed)
    model = CoefficientSPDE(data.grid, data.projection, history=args.history,
                            width=args.width, depth=args.depth, neural=role == "neural").to(device)
    count = sum(p.numel() for p in model.parameters())
    checkpoint, info_path = out/f"{role}_best.pt", out/f"{role}_training.json"
    if args.resume and checkpoint.exists() and info_path.exists():
        info = json.loads(info_path.read_text())
        if info.get("finished"):
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            print(f"Using completed {role}: {count:,} parameters", flush=True)
            return model, info
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003 if role == "neural" else .008, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)
    origins = data.origins("train", history=args.history, horizon=24)
    log, best, stale = [], float("inf"), 0
    started = time.monotonic()
    print(f"Training {role}: {count:,} parameters; {len(origins):,} overlapping training origins", flush=True)
    for step in range(1, args.steps+1):
        model.train()
        chosen = rng.choice(origins, args.batch)
        batch = data.batch(chosen, args.history, 1, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.negative_log_likelihood(batch["history"], batch["calendar"][:, 0], batch["target"][:, 0], batch["mask"][:, 0])
        likelihood = float(loss.detach())
        auxiliary = 0.
        if step % 4 == 0:
            horizon = 24 if step % 20 == 0 else 6
            selected = rng.choice(origins, 4 if horizon == 24 else 8)
            window = data.batch(selected, args.history, horizon, device)
            paths = model.rollout(window["history"], window["calendar"], window["daylight"], members=4)
            extra = rollout_loss(paths, window["target"], window["mask"])
            loss = loss+.5*extra
            auxiliary = float(extra.detach())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at {role}/{step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(f"  {role} step {step}/{args.steps}, NLL={likelihood:.4f}, rollout={auxiliary:.4f}, seconds={time.monotonic()-started:.0f}", flush=True)
        if step == 1 or step % args.check_every == 0 or step == args.steps:
            model.eval()
            score = validate(model, data, device, scales, seed=args.seed+300)
            improved = score < best-1e-4
            if improved:
                best, stale = score, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
            log.append({"step": step, "training_nll": likelihood, "rollout_loss": auxiliary,
                        "validation_crps": score, "best_validation_crps": best, "elapsed_seconds": time.monotonic()-started})
            info = {"role": role, "parameters": count, "best_validation_crps": best,
                    "finished": False, "history": log}
            write_json(info_path, info)
            print(f"  {role} validation CRPS={score:.4f}, best={best:.4f}", flush=True)
            if stale >= args.patience:
                print(f"  {role}: validation early stopping", flush=True)
                break
    info.update(finished=True, steps_completed=step, training_seconds=time.monotonic()-started)
    write_json(info_path, info)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model, info


@torch.no_grad()
def export_weather(model, data, args, device, out):
    """Keep the original July scenario stream; comparisons use benchmark_spde.py."""
    model.eval()
    torch.manual_seed(args.seed+500)
    selected = monthly_origins(data, "test", args.test_hours)
    # Advance through the same first seven origins as the original evaluator.
    # This preserves the existing saved July weather, including its random stream.
    for origin in selected[:7]:
        batch = data.batch([origin], args.history, args.test_hours, device)
        paths = model.rollout(batch["history"], batch["calendar"], batch["daylight"], members=args.members)[0].cpu().numpy()
        if origin != selected[6]:
            continue
        indices = batch["indices"][0]
        sample = physical_samples(data, paths, indices)
        np.savez_compressed(out/f"neural_weather_{args.test_hours}h.npz", latent=paths,
            raw=paths*data.scale[None, None, :, None]+data.background[indices][None],
            timestamps=data.times[indices].asi8, grid_x_m=data.grid.x, grid_y_m=data.grid.y,
            site_ids=data.sites.site_id.to_numpy(str), field_order=np.array(data.manifest["field_order"]),
            source_manifest_sha256=np.array(digest(data.directory/"manifest.json")), **sample)
        print(f"Saved neural weather: {out/f'neural_weather_{args.test_hours}h.npz'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/processed/spde/multiyear_2015_2019")
    parser.add_argument("--output", type=Path, default=ROOT/"output/spde/neural_v1")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--members", type=int, default=48)
    parser.add_argument("--test-hours", type=int, default=72)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.width, args.depth, args.history, args.steps, args.batch, args.check_every, args.patience) < 1 or args.members < 2 or args.test_hours < 24:
        parser.error("Use positive model/training settings, >=2 members and >=24 test hours.")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError("Choose a new output directory or --resume matching completed models.")
    out.mkdir(parents=True, exist_ok=True)
    context = {"data_manifest_sha256": digest(args.data/"manifest.json"), "width": args.width,
               "depth": args.depth, "history": args.history, "steps": args.steps, "seed": args.seed}
    if args.resume and (out/"protocol.json").exists():
        old = json.loads((out/"protocol.json").read_text())
        if old["context"] != context:
            raise ValueError("Resume configuration differs; use a new output directory.")
    data = MultiYearWeather(args.data)
    if len(monthly_origins(data, "validation", 24)) != 12*len(data.manifest["split"]["validation"]) or len(monthly_origins(data, "test", args.test_hours)) != 12*len(data.manifest["split"]["test"]):
        raise ValueError("Validation/test data must cover every prescribed monthly origin and complete rollout.")
    print(f"Loaded {len(data.times):,} hourly fields, training years {data.manifest['split']['train']}", flush=True)
    scales = physical_scales(data)
    np.savez_compressed(out/"preprocessing.npz", coefficients=data.coefficients, scale=data.scale, projection=data.projection)
    sources = {name: digest(ROOT/name) for name in ("src/neural_weather_spde.py", "data/generate_data/train_neural_spde.py", "data/generate_data/neural_spde_data.py")}
    training_sources = old.get("training_source_sha256", old["source_sha256"]) if args.resume and (out/"protocol.json").exists() else sources
    protocol = {"context": context, "split": data.manifest["split"], "held_sites": data.held.tolist(),
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "physical_scales": scales,
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": sources, "training_source_sha256": training_sources}
    write_json(out/"protocol.json", protocol)
    models, info = {}, {}
    for role in ("constant", "neural"):
        models[role], info[role] = train_one(role, data, args, device, scales, out)
    export_weather(models["neural"], data, args, device, out)
    print("Training/weather export complete. Use data/generate_data/benchmark_spde.py for unified comparisons.", flush=True)


if __name__ == "__main__":
    main()
