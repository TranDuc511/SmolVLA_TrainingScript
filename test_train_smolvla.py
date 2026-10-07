"""Wrapper checks that run without LeRobot, model downloads, or a GPU."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import train_smolvla


class EvaluationTests(unittest.TestCase):
    def dry_run(self, *args):
        console = io.StringIO()
        with patch.object(sys, "argv", ["train_smolvla.py", "--dry-run", *args]):
            with contextlib.redirect_stdout(console):
                self.assertEqual(train_smolvla.main(), 0)
        return console.getvalue()

    def test_new_run_defaults_and_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new_run"
            text = self.dry_run("--output-dir", str(output))
            self.assertIn("--dataset.eval_split=0.1", text)
            self.assertIn("--eval_steps=10000", text)
            self.assertIn("--max_eval_samples=1000", text)
            self.assertFalse(output.exists())

    def test_disabled_split_and_invalid_settings(self):
        self.assertIn("--eval_steps=0", self.dry_run("--eval-split", "0"))
        for args in [("--eval-split", "0", "--eval-freq", "5"),
                     ("--eval-split", "1"), ("--eval-split", "nan"),
                     ("--eval-freq", "-1"), ("--max-eval-samples", "-1")]:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.dry_run(*args)
            self.assertEqual(error.exception.code, 2)

    def test_resume_preserves_settings_and_resolves_log_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path = Path(folder) / "train_config.json"
            output = Path(folder) / "original"
            config_path.write_text(json.dumps({
                "output_dir": str(output), "policy": {"device": "cpu"},
                "dataset": {"eval_split": 0.2}, "eval_steps": 50,
            }), encoding="utf-8")
            text = self.dry_run("--resume", str(config_path))
            self.assertNotIn("--dataset.eval_split=", text)
            self.assertNotIn("--eval_steps=", text)
            self.assertIn(str(output / "evaluation"), text)
            override = Path(folder) / "override"
            text = self.dry_run("--resume", str(config_path), "--output-dir", str(override),
                                "--eval-freq", "25")
            self.assertIn("--eval_steps=25", text)
            self.assertIn(str(override / "evaluation"), text)
            # Legacy checkpoints must not silently gain a holdout on resume.
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["dataset"] = {}
            config.pop("eval_steps")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.assertNotIn("--dataset.eval_split=", self.dry_run("--resume", str(config_path)))

    def test_results_append_console_stays_visible_and_exit_code_propagates(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            code = (
                "import sys; from pathlib import Path; "
                f"assert not Path({str(output)!r}).exists(); "
                "print('training loss=9'); "
                "print('INFO step 5: eval_loss=0.1234', file=sys.stderr); "
                "sys.exit(7)"
            )
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                self.assertEqual(train_smolvla.run_training([sys.executable, "-c", code], output), 7)
                self.assertEqual(train_smolvla.run_training([
                    sys.executable, "-c", "print('INFO step 10: eval_loss=0.1000')"
                ], output), 0)
            self.assertIn("training loss=9", console.getvalue())
            self.assertIn("eval_loss=0.1234", console.getvalue())
            with (output / "evaluation" / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["step"] for row in rows], ["5", "10"])
            self.assertEqual(float(rows[0]["eval_loss"]), 0.1234)
            log = (output / "evaluation" / "evaluation.log").read_text(encoding="utf-8")
            self.assertEqual(log.count("eval_loss="), 2)
            self.assertNotIn("training loss", log)

    def test_no_eval_results_creates_no_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(train_smolvla.run_training([
                    sys.executable, "-c", "print('training only')"
                ], output), 0)
            self.assertFalse(output.exists())

    def test_successful_training_evaluates_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            calls = []

            def run(command, run_output):
                calls.append(command)
                if len(calls) == 1:
                    for step in (5, 10):
                        checkpoint = output / "checkpoints" / str(step) / "pretrained_model"
                        checkpoint.mkdir(parents=True)
                        (checkpoint / "train_config.json").write_text("{}", encoding="utf-8")
                return 0

            with patch.object(sys, "argv", ["train_smolvla.py", "--device", "cpu", "--steps", "10",
                                           "--output-dir", str(output), "--max-eval-samples", "2"]), \
                 patch.object(sys, "version_info", (3, 12)), \
                 patch.object(train_smolvla, "version", return_value="0.6.1"), \
                 patch.object(train_smolvla, "run_training", side_effect=run), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(train_smolvla.main(), 0)
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[1][1].endswith("evaluate_smolvla.py"))
            self.assertIn(str(output / "checkpoints" / "10" / "pretrained_model"), calls[1])
            self.assertEqual(calls[1][-2:], ["--max-eval-samples", "2"])

    def test_failed_or_disabled_training_skips_checkpoint_evaluation(self):
        for code, extra in [(7, []), (0, ["--eval-freq", "0"])]:
            with tempfile.TemporaryDirectory() as folder:
                with patch.object(sys, "argv", ["train_smolvla.py", "--device", "cpu",
                                               "--output-dir", str(Path(folder) / "run"), *extra]), \
                     patch.object(sys, "version_info", (3, 12)), \
                     patch.object(train_smolvla, "version", return_value="0.6.1"), \
                     patch.object(train_smolvla, "run_training", return_value=code) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(train_smolvla.main(), code)
                    self.assertEqual(run.call_count, 1)

    def test_training_timing_appends_sessions_and_records_interruption(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            with patch.object(train_smolvla, "run_training", return_value=0), \
                 patch.object(train_smolvla.time, "perf_counter", side_effect=[10, 75, 100, 105]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(train_smolvla.timed_training([], output, False), 0)
                self.assertEqual(train_smolvla.timed_training([], output, True), 0)
            with patch.object(train_smolvla, "run_training", side_effect=KeyboardInterrupt), \
                 patch.object(train_smolvla.time, "perf_counter", side_effect=[200, 202]), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                train_smolvla.timed_training([], output, True)
            rows = [json.loads(line) for line in (output / "training_timing.jsonl").read_text().splitlines()]
            self.assertEqual([row["duration_seconds"] for row in rows], [65, 5, 2])
            self.assertEqual(rows[0]["duration"], "0:01:05")
            self.assertEqual([row["resumed"] for row in rows], [False, True, True])
            self.assertEqual(rows[-1]["status"], "interrupted")
            self.assertEqual(rows[-1]["exit_code"], 130)

    def test_startup_failure_timing_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            console = io.StringIO()
            with patch.object(train_smolvla, "run_training", return_value=7), contextlib.redirect_stdout(console):
                self.assertEqual(train_smolvla.timed_training([], output, False), 7)
            self.assertFalse(output.exists())
            self.assertIn("Training duration:", console.getvalue())


if __name__ == "__main__":
    unittest.main()
