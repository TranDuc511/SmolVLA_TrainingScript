#!/usr/bin/env python
"""Fine-tune SmolVLA on Project-IRA's merged SO-101 dataset using LeRobot."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, help="total updates (default: 200000)")
    parser.add_argument("--batch-size", type=positive_int, help="default: 8; lower if GPU memory is limited")
    parser.add_argument("--output-dir", type=Path, help=f"new run directory (default: {OUTPUT})")
    parser.add_argument("--num-workers", type=int, help="default: 0 on Windows, 4 elsewhere")
    parser.add_argument("--save-freq", type=positive_int, help="checkpoint interval (default: 10000)")
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

    # On resume, unspecified options retain the checkpoint's configuration.
    options = {
        "steps": (args.steps, 200_000),
        "batch_size": (args.batch_size, 8),
        "num_workers": (args.num_workers, 0 if sys.platform == "win32" else 4),
        "save_freq": (args.save_freq, 10_000),
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
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
