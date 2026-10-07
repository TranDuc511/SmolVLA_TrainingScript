#!/usr/bin/env python
"""Evaluate a SmolVLA checkpoint on its saved held-out episode split."""

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import psutil

from train_smolvla import nonnegative_int


class ActionErrors:
    """Aggregate over valid scalar action entries, rather than batch means."""

    def __init__(self):
        self.absolute = None
        self.squared = None
        self.count = 0

    def update(self, prediction, target, valid):
        if prediction.shape != target.shape or valid.shape != target.shape[:-1]:
            raise ValueError("prediction, target, and padding mask shapes must match")
        error = (prediction.astype(np.float64) - target.astype(np.float64))[valid]
        if not np.isfinite(error).all():
            raise ValueError("non-finite action prediction or target")
        absolute = np.abs(error).sum(axis=0)
        squared = np.square(error).sum(axis=0)
        self.absolute = absolute if self.absolute is None else self.absolute + absolute
        self.squared = squared if self.squared is None else self.squared + squared
        self.count += error.shape[0]

    def result(self):
        if not self.count:
            raise ValueError("evaluation contains no valid action entries")
        return {
            "mae": float(self.absolute.sum() / (self.count * len(self.absolute))),
            "rmse": float(np.sqrt(self.squared.sum() / (self.count * len(self.squared)))),
            "mae_per_joint": (self.absolute / self.count).tolist(),
            "rmse_per_joint": np.sqrt(self.squared / self.count).tolist(),
            "valid_action_timesteps": self.count,
        }


class RamMonitor:
    """Sample evaluator RSS, including its resident model and dataset."""

    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def select_indices(dataset, budget):
    """Select evenly spaced frames per task, with a strict total budget."""
    tasks = np.asarray(dataset.hf_dataset.data.column("task_index").to_numpy())
    groups = [np.flatnonzero(tasks == task) for task in np.unique(tasks)]
    if budget == 0 or budget >= len(tasks):
        return list(range(len(tasks)))
    counts = [0] * len(groups)
    remaining = budget
    while remaining:
        for i, group in enumerate(groups):
            if counts[i] < len(group) and remaining:
                counts[i] += 1
                remaining -= 1
    return sorted(int(group[index]) for group, count in zip(groups, counts)
                  for index in np.linspace(0, len(group) - 1, count, dtype=int))


def evaluate(policy, preprocessor, postprocessor, dataset, indices, folder, device, use_amp):
    """Run batch-size-one loss and chunk inference; save each chunk immediately."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from lerobot.utils.collate import lerobot_collate_fn

    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=0,
                        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None)
    errors = ActionErrors()
    latencies = []
    loss_sum = 0.0
    loss_weight = 0
    policy.eval()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def autocast():
        return torch.autocast(device_type=device.type) if use_amp else nullcontext()

    predictions_dir = folder / "inference"
    predictions_dir.mkdir()
    warmed_up = False
    with RamMonitor() as memory, torch.inference_mode():
        for sample_number, raw in enumerate(loader):
            target = raw["action"].cpu().numpy().copy()
            valid = (~raw["action_is_pad"].bool()).cpu().numpy() if "action_is_pad" in raw else np.ones(target.shape[:-1], dtype=bool)
            for key in dataset.meta.camera_keys:
                if key in raw and raw[key].dtype == torch.uint8:
                    raw[key] = raw[key].float() / 255.0
            batch = preprocessor(raw)
            # Warm up one full chunk, excluding it from latency measurements.
            if not warmed_up:
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []), autocast():
                    policy.reset()
                    policy.predict_action_chunk({k: v for k, v in batch.items() if k != "action"})
                sync()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                warmed_up = True

            # Loss is SmolVLA's native flow-matching objective. Weight by the
            # valid elements its forward() uses, excluding episode padding.
            with autocast():
                loss, _ = policy.forward(dict(batch))
            weight = int(valid.sum()) * target.shape[-1]
            value = float(loss.item())
            if not np.isfinite(value):
                raise ValueError("non-finite flow-matching loss")
            loss_sum += value * weight
            loss_weight += weight

            policy.reset()
            sync()
            start = time.perf_counter()
            with autocast():
                predicted = policy.predict_action_chunk({k: v for k, v in batch.items() if k != "action"})
            sync()
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            # Saved checkpoint unnormalization restores dataset action units.
            prediction = postprocessor(predicted).detach().float().cpu().numpy()
            errors.update(prediction, target, valid)
            ids = {key: raw[key].cpu().numpy() for key in
                   ("episode_index", "frame_index", "index", "task_index", "timestamp") if key in raw}
            np.savez_compressed(predictions_dir / f"sample_{sample_number:06d}.npz",
                                predicted_actions=prediction, target_actions=target,
                                valid_action_mask=valid, latency_ms=latency_ms,
                                flow_matching_loss=value, **ids)
            print(f"Inference {sample_number + 1}/{len(indices)}: {latency_ms:.2f} ms", flush=True)

    result = errors.result()
    result.update({
        "flow_matching_loss": loss_sum / loss_weight,
        "samples": len(latencies), "inference_batch_size": 1,
        "action_chunk_length": prediction.shape[1],
        "chunk_latency_mean_ms": float(np.mean(latencies)),
        "chunk_latency_p50_ms": float(np.percentile(latencies, 50)),
        "chunk_latency_p95_ms": float(np.percentile(latencies, 95)),
        "ram_rss_end_mib": memory.process.memory_info().rss / 2**20,
        "ram_rss_sampled_peak_mib": memory.peak / 2**20,
        "vram_allocated_end_mib": torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "vram_reserved_end_mib": torch.cuda.memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "vram_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "vram_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="checkpoint's pretrained_model directory")
    parser.add_argument("--output-dir", type=Path, help="evaluation root (default: saved run/evaluation)")
    parser.add_argument("--max-eval-samples", type=nonnegative_int, help="strict frame budget (default: checkpoint setting; 0 uses all)")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "train_config.json").is_file():
        parser.error("checkpoint must contain train_config.json")
    from importlib.metadata import version
    from train_smolvla import LEROBOT_VERSION
    if version("lerobot") != LEROBOT_VERSION:
        parser.error(f"requires LeRobot {LEROBOT_VERSION}")
    import torch
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.random_utils import set_seed

    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    if cfg.dataset.eval_split <= 0:
        parser.error("checkpoint has no held-out split; train a new run with --eval-split > 0")
    policy_cfg = SmolVLAConfig.from_pretrained(checkpoint)
    device = torch.device(args.device or policy_cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; pass --device cpu")
    policy_cfg.device = str(device)
    cfg.policy = policy_cfg
    set_seed(args.seed)
    train_dataset, dataset = make_train_eval_datasets(cfg)
    train_episodes = train_dataset.episodes
    del train_dataset
    budget = cfg.max_eval_samples if args.max_eval_samples is None else args.max_eval_samples
    indices = select_indices(dataset, budget)
    if not indices:
        parser.error("held-out dataset is empty")
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=policy_cfg).to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)},
                                "rename_observations_processor": {"rename_map": cfg.rename_map}},
    )
    root = (args.output_dir or Path(cfg.output_dir) / "evaluation").expanduser().resolve()
    now = datetime.now(timezone.utc)
    folder = root / f"checkpoint_{checkpoint.parent.name}_{now:%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    metadata = {
        "checkpoint": str(checkpoint), "timestamp_utc": now.isoformat(),
        "seed": args.seed, "device": str(device), "use_amp": policy_cfg.use_amp and device.type == "cuda",
        "dataset_repo_id": cfg.dataset.repo_id, "dataset_revision": cfg.dataset.revision,
        "eval_split": cfg.dataset.eval_split, "train_episodes": train_episodes,
        "eval_episodes": dataset.episodes, "selected_dataset_indices": indices,
        "action_names": dataset.meta.features["action"].get("names"),
        "action_units": "dataset units; checkpoint postprocessor unnormalization",
        "latency_scope": "batch-size-one predict_action_chunk only; CUDA synchronized; one warmup excluded",
        "memory_scope": "evaluator RSS sampled every 10 ms; CUDA allocator includes loss and inference after warmup",
        "lerobot_version": LEROBOT_VERSION, "torch_version": torch.__version__,
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = evaluate(policy, preprocessor, postprocessor, dataset, indices, folder, device, metadata["use_amp"])
    (folder / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    row = {"timestamp_utc": now.isoformat(), "checkpoint": str(checkpoint), "results_dir": str(folder),
           **{key: value for key, value in result.items() if not isinstance(value, list)}}
    csv_path = root / "checkpoint_metrics.csv"
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(result, indent=2))
    print(f"Saved evaluation: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
