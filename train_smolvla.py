#!/usr/bin/env python
"""Fine-tune SmolVLA on Project-IRA's merged SO-101 dataset using LeRobot."""

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import subprocess
import sys
import time

DATASET = "Project-IRA/TPSoSe2026_Dataset_Full_Merged_Final_LeRobot_SO101_V1"
OUTPUT = "outputs/train/smolvla_so101"
LEROBOT_VERSION = "0.6.1"
CAMERAS = {
    "observation.images.desk_view": "observation.images.camera1",
    "observation.images.wrist_left": "observation.images.camera2",
}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def eval_split(value: str) -> float:
    fraction = float(value)
    if not 0 <= fraction < 1:
        raise argparse.ArgumentTypeError("must be at least 0 and less than 1")
    return fraction


def run_training(command: list[str], output: Path) -> int:
    """Keep console output visible and append LeRobot's eval results locally."""
    # Create the folder only once evaluation starts: LeRobot rejects an existing
    # output directory for a new run during its initial config validation.
    pattern = re.compile(r"\bstep (\d+): eval_loss=([^\s]+)")
    with ExitStack() as stack:
        log_file = None
        writer = None
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        stack.callback(process.stdout.close)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                match = pattern.search(line)
                if match is None:
                    continue
                loss = float(match[2])
                if log_file is None:
                    folder = output / "evaluation"
                    folder.mkdir(parents=True, exist_ok=True)
                    log_file = stack.enter_context((folder / "evaluation.log").open("a", encoding="utf-8"))
                    csv_path = folder / "metrics.csv"
                    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
                    csv_file = stack.enter_context(csv_path.open("a", encoding="utf-8", newline=""))
                    writer = csv.writer(csv_file)
                    if needs_header:
                        writer.writerow(["timestamp_utc", "step", "eval_loss"])
                log_file.write(line)
                log_file.flush()
                writer.writerow([datetime.now(timezone.utc).isoformat(), int(match[1]), loss])
                csv_file.flush()
            return process.wait()
        finally:
            # Do not leave a training child running after an interrupt or log error.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def timed_training(command: list[str], output: Path, resumed: bool) -> int:
    """Measure the training subprocess only; append one record per session."""
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    exit_code = None
    try:
        exit_code = run_training(command, output)
        return exit_code
    except KeyboardInterrupt:
        exit_code = 130
        raise
    finally:
        elapsed = time.perf_counter() - started
        duration = str(timedelta(seconds=round(elapsed)))
        record = {
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": elapsed,
            "duration": duration,
            "resumed": resumed,
            "exit_code": exit_code,
            "status": "completed" if exit_code == 0 else "interrupted" if exit_code == 130 else "failed",
            "scope": "training process wall time; includes setup, downloads, periodic evaluation and checkpoints; excludes final checkpoint evaluation",
        }
        print(f"Training duration: {duration} ({elapsed:.2f} seconds)", flush=True)
        # A startup failure should not create a run directory that prevents retry.
        if output.is_dir() or exit_code == 0:
            output.mkdir(parents=True, exist_ok=True)
            path = output / "training_timing.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"Training timing saved: {path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, help="total updates (default: 200000)")
    parser.add_argument("--batch-size", type=positive_int, help="default: 8; lower if GPU memory is limited")
    parser.add_argument("--output-dir", type=Path, help=f"new run directory (default: {OUTPUT})")
    parser.add_argument("--num-workers", type=int, help="default: 0 on Windows, 4 elsewhere")
    parser.add_argument("--save-freq", type=positive_int, help="checkpoint interval (default: 10000)")
    parser.add_argument("--eval-freq", type=nonnegative_int, help="held-out loss interval (default: 10000; 0 disables)")
    parser.add_argument("--eval-split", type=eval_split, help="held-out episode fraction (default: 0.1; 0 disables)")
    parser.add_argument("--max-eval-samples", type=nonnegative_int, help="evaluation sample budget across tasks (default: 1000; 0 uses all)")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="default: cuda")
    parser.add_argument("--amp", action="store_true", help="enable mixed precision (CUDA only)")
    parser.add_argument("--resume", type=Path, metavar="TRAIN_CONFIG", help="checkpoint's train_config.json")
    parser.add_argument("--dry-run", action="store_true", help="print arguments without downloads or training")
    args = parser.parse_args()
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.amp and args.device == "cpu":
        parser.error("--amp requires CUDA")

    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    if args.resume:
        config_path = args.resume.expanduser().resolve()
        if not config_path.is_file():
            parser.error(f"resume config does not exist: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        output = (args.output_dir or Path(config["output_dir"])).expanduser().resolve()
        device = args.device or config["policy"]["device"]
        if args.amp and device != "cuda":
            parser.error("--amp requires a CUDA checkpoint or --device cuda")
        command += [f"--config_path={config_path}", "--resume=true"]
    else:
        device = args.device or "cuda"
        output = (args.output_dir or Path(OUTPUT)).expanduser().resolve()
        if output.exists():
            parser.error(f"output already exists: {output}; choose --output-dir or --resume")
        command += [
            f"--dataset.repo_id={DATASET}",
            "--dataset.video_backend=pyav",
            "--policy.path=lerobot/smolvla_base",
            f"--policy.device={device}",
            "--policy.empty_cameras=1",
            f"--rename_map={json.dumps(CAMERAS, separators=(',', ':'))}",
            f"--output_dir={output}",
            "--job_name=smolvla_so101",
            "--seed=42",
            "--env_eval_freq=0",
            "--log_freq=100",
            "--save_checkpoint=true",
            "--wandb.enable=false",
        ]

    split = args.eval_split if args.eval_split is not None else (config["dataset"].get("eval_split", 0.0) if args.resume else 0.1)
    eval_freq = args.eval_freq if args.eval_freq is not None else (config.get("eval_steps", 0) if args.resume else 10_000)
    if split == 0 and args.eval_freq is None:
        eval_freq = 0
    if eval_freq > 0 and split == 0:
        parser.error("--eval-freq requires --eval-split greater than 0")
    if args.eval_split is not None or not args.resume:
        command.append(f"--dataset.eval_split={split}")
    if args.eval_freq is not None or args.eval_split is not None or not args.resume:
        command.append(f"--eval_steps={eval_freq}")

    # On resume, unspecified options retain the checkpoint's configuration.
    options = {
        "steps": (args.steps, 200_000),
        "batch_size": (args.batch_size, 8),
        "num_workers": (args.num_workers, 0 if sys.platform == "win32" else 4),
        "save_freq": (args.save_freq, 10_000),
        "max_eval_samples": (args.max_eval_samples, 1_000),
    }
    for name, (value, default) in options.items():
        if value is not None or not args.resume:
            command.append(f"--{name}={default if value is None else value}")
    if args.resume and args.output_dir:
        command.append(f"--output_dir={args.output_dir.expanduser().resolve()}")
    if args.resume and args.device:
        command.append(f"--policy.device={args.device}")
    if args.amp:
        command.append("--policy.use_amp=true")
    command += ["--policy.push_to_hub=false", "--save_checkpoint_to_hub=false"]

    print("Training arguments:\n" + "\n".join(command[3:]), flush=True)
    print(f"Evaluation results: {output / 'evaluation'}", flush=True)
    if args.dry_run:
        return 0
    if sys.version_info < (3, 12):
        parser.error("LeRobot 0.6.1 requires Python 3.12+. See README.md.")
    try:
        installed = version("lerobot")
    except PackageNotFoundError:
        parser.error("install dependencies first: python -m pip install -r requirements.txt")
    if installed != LEROBOT_VERSION:
        parser.error(f"expected LeRobot {LEROBOT_VERSION}, found {installed}; install requirements.txt")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install CUDA-enabled PyTorch or use --device cpu")
    try:
        result = timed_training(command, output, resumed=bool(args.resume))
        if result != 0 or eval_freq == 0:
            return result
        # Training has exited, so its model and optimizer no longer occupy RAM/VRAM.
        checkpoints = [path for path in (output / "checkpoints").glob("*/pretrained_model")
                       if path.parent.name.isdigit() and (path / "train_config.json").is_file()]
        if not checkpoints:
            parser.error("training completed but no saved checkpoint was found for evaluation")
        checkpoint = max(checkpoints, key=lambda path: int(path.parent.name))
        evaluation_command = [sys.executable, str(Path(__file__).with_name("evaluate_smolvla.py")),
                              "--checkpoint", str(checkpoint), "--output-dir", str(output / "evaluation"),
                              "--device", device]
        if args.max_eval_samples is not None:
            evaluation_command += ["--max-eval-samples", str(args.max_eval_samples)]
        print("Evaluating saved checkpoint with action metrics and inference profiling...", flush=True)
        return run_training(evaluation_command, output)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
