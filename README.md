# SmolVLA SO-101 training

Fine-tune `lerobot/smolvla_base` on [Project-IRA's merged dataset](https://huggingface.co/datasets/Project-IRA/TPSoSe2026_Dataset_Full_Merged_Final_LeRobot_SO101_V1): 930 episodes, two cameras, six joint values, and recorded task prompts. LeRobot loads the data and normalization statistics automatically.

## Setup

Use Python 3.12 and an NVIDIA GPU; Linux/WSL2 is recommended. Run from this directory in a terminal with Conda available:

```sh
conda create -n smolvla python=3.12 -y
conda activate smolvla
conda install -c conda-forge ffmpeg=7.1.1 -y
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

The CUDA check must print `True` for GPU training. See [LeRobot installation](https://huggingface.co/docs/lerobot/installation) for driver/CUDA alternatives. Allow space for the roughly 9.3 GB dataset, model downloads, and multiple checkpoints. Public downloads need no login; use `hf auth login` if authentication is requested.

## Train

```sh
python train_smolvla.py --dry-run
python train_smolvla.py --steps 10 --batch-size 2 --output-dir outputs/train/smoke
python train_smolvla.py
```

The smoke run checks downloading, video decoding, training, and checkpoint saving. Defaults: **200,000 steps**, **batch size 8**, checkpoint every **10,000 steps**, seed 42. Optional example:

```sh
python train_smolvla.py --batch-size 32 --amp --output-dir outputs/train/run2
```

Reduce `--batch-size` if CUDA runs out of memory. Use `--num-workers 0` for loader issues (already the Windows default); CPU training via `--device cpu` is slow. On native Windows, enable Developer Mode for LeRobot's checkpoint symlinks. `--help` lists options.

The script [maps cameras](https://huggingface.co/docs/lerobot/rename_map) from `desk_view` to `camera1` and `wrist_left` to `camera2`, and masks an unused camera input. It uses PyAV decoding and LeRobot's SmolVLA optimizer settings. Checkpoints stay local; Hub uploads and W&B are disabled.

## Checkpoints and resume

Final model: `outputs/train/smolvla_so101/checkpoints/200000/pretrained_model/` (the step folder matches your chosen total). Each checkpoint also includes optimizer/scheduler state for resuming:

```sh
python train_smolvla.py --resume outputs/train/smolvla_so101/checkpoints/010000/pretrained_model/train_config.json
```

Resume retains saved settings; optionally pass `--steps` to increase the **total** target. Keep the entire checkpoint folder. New runs require a fresh output directory.

## Evaluation logs

Training prints its elapsed duration and saves a record in `<output-dir>/training_timing.jsonl`. Each invocation appends its UTC start/end times, duration in seconds and readable form, completion status, exit code, and whether it resumed a checkpoint. Timing covers the training process, including setup/downloads, periodic evaluation, and checkpoint saving; the final checkpoint evaluation is excluded. Resumed sessions get separate records rather than overwriting earlier timings. Interrupted/failed sessions are recorded if the run directory exists; a failure before the run directory is created prints the duration without creating a directory. Dry runs write no timing files.

New runs hold out 10% of episodes per task using LeRobot's native split (the last episodes of each task, rounded up). These episodes are excluded from training. Every 10,000 updates, LeRobot computes held-out loss using a sample budget of 1,000 frames shared across tasks. The budget is approximate: LeRobot selects at least one frame per task. Evaluation adds runtime; adjust the interval and sample budget as needed:

```sh
python train_smolvla.py --eval-freq 5000 --eval-split 0.1 --max-eval-samples 2000
python train_smolvla.py --steps 10 --eval-freq 5 --batch-size 2 --output-dir outputs/train/eval_smoke
```

Results appear at the first evaluation under the run's output directory:

```text
outputs/train/smolvla_so101/evaluation/
  evaluation.log  # evaluation console messages
  metrics.csv     # timestamp_utc, step, eval_loss
```

Console output remains visible. Files are flushed after each evaluation and appended on resume, including repeated steps if resuming an earlier checkpoint. Loss values use LeRobot's logged precision (four decimal places). The first evaluation occurs at a multiple of `--eval-freq`; set the interval at or below `--steps` for short runs. LeRobot does not add an extra evaluation at the final step.

### Checkpoint metrics and saved inference

After successful training, when evaluation is enabled, the wrapper evaluates the latest saved checkpoint in a separate process. This releases the training model and optimizer before inference profiling. This final pass also runs for short jobs that finish before the first periodic evaluation. Periodic `metrics.csv` continues to contain only LeRobot's loss; the additional metrics are written to `checkpoint_metrics.csv` and a unique folder per checkpoint evaluation:

```text
evaluation/
  evaluation.log
  metrics.csv
  checkpoint_metrics.csv
  checkpoint_200000_<timestamp>_<id>/
    metadata.json
    summary.json
    inference/
      sample_000000.npz
      sample_000001.npz
      ...
```

You can also evaluate an existing checkpoint directly:

```sh
python evaluate_smolvla.py --checkpoint outputs/train/smolvla_so101/checkpoints/200000/pretrained_model --max-eval-samples 1000
```

`--device cpu` overrides the checkpoint device. The evaluator requires the checkpoint's saved held-out split and processors. It never creates a new split for an already trained checkpoint. Each evaluation uses seed 42 by default (`--seed` overrides it), batch size one, and evenly spaced frames per task. Its `--max-eval-samples` is a strict total budget, defaulting to the saved checkpoint setting. Use 0 for all held-out frames.

| Metric | Definition |
| --- | --- |
| MAE | Mean absolute prediction error across valid action timesteps and joints, in dataset action units after checkpoint unnormalization. |
| RMSE | Square root of mean squared prediction error over the same valid entries. Per-joint MAE/RMSE are included in `summary.json`. |
| Flow-matching loss | SmolVLA's native `forward()` objective on normalized actions, weighted by valid scalar action entries. This is a separate stochastic forward pass from periodic evaluation, so its value can differ. |
| Action-chunk latency | Mean, median (p50), and p95 milliseconds for batch-size-one `predict_action_chunk()`, with one warmup excluded and CUDA synchronized before/after each timed call. Excludes dataset loading, preprocessing, unnormalization, and file writes. |
| RAM | Evaluator process RSS at completion and sampled peak (every 10 ms), including the resident model and dataset. Brief peaks between samples can be missed. |
| VRAM | PyTorch allocated/reserved memory at completion and allocator peaks during the loss/inference pass after warmup, in MiB. CPU evaluations report null/blank VRAM. These are evaluator measurements, not training memory or total device usage. |

Each compressed NumPy inference file contains `predicted_actions` and `target_actions` (shape `1 × chunk_length × action_dim`), `valid_action_mask`, chunk latency, flow-matching loss, and available episode/frame/task/index/timestamp identifiers. Padded predictions are saved but excluded from metrics. Load with `numpy.load(path, allow_pickle=False)`. Metadata records the checkpoint, split episodes, selected frames, seed, device, action names, and measurement scope. A `summary.json` and CSV row are written only after the pass completes; an interrupted pass may leave partial inference files. Inference sampling and file writes add runtime and disk usage.

Use `--max-eval-samples 0` to evaluate all held-out frames. `--eval-freq 0` disables periodic and automatic final evaluation while retaining the holdout; `--eval-split 0` trains on all episodes and disables evaluation unless a positive `--eval-freq` is explicitly passed (an error).

Resume preserves the checkpoint's evaluation settings. Old checkpoints without a holdout remain evaluation-disabled. Changing `--eval-split` during resume changes the training data and can evaluate episodes already seen by that checkpoint; start a fresh run for a clean held-out comparison.

Held-out loss does not measure robot task success. Evaluate checkpoints on the target setup with the same joint order and camera mapping. Dataset attribution: Project-IRA, CC BY-SA 4.0; see its card for terms. Training follows [official SmolVLA guidance](https://huggingface.co/docs/lerobot/smolvla).
