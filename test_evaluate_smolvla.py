"""Numerical and inference artifact checks using a tiny CPU policy."""

import contextlib
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from evaluate_smolvla import ActionErrors, evaluate, select_indices


class MetricTests(unittest.TestCase):
    def test_padding_and_scalar_weighting(self):
        errors = ActionErrors()
        errors.update(np.array([[[1., 3.], [100., 100.]]]), np.zeros((1, 2, 2)),
                      np.array([[True, False]]))
        errors.update(np.array([[[2., 4.], [0., 6.]]]), np.zeros((1, 2, 2)),
                      np.array([[True, True]]))
        result = errors.result()
        self.assertAlmostEqual(result["mae"], 16 / 6)
        self.assertAlmostEqual(result["rmse"], np.sqrt(66 / 6))
        np.testing.assert_allclose(result["mae_per_joint"], [1, 13 / 3])
        self.assertEqual(result["valid_action_timesteps"], 3)

    def test_empty_and_nonfinite_actions_fail(self):
        with self.assertRaises(ValueError):
            ActionErrors().result()
        with self.assertRaises(ValueError):
            ActionErrors().update(np.array([[[np.nan]]]), np.zeros((1, 1, 1)), np.ones((1, 1), dtype=bool))

    def test_task_selection_respects_budget_and_spans_frames(self):
        tasks = np.array([0] * 10 + [1] * 3)
        column = types.SimpleNamespace(to_numpy=lambda: tasks)
        dataset = types.SimpleNamespace(hf_dataset=types.SimpleNamespace(
            data=types.SimpleNamespace(column=lambda name: column)))
        self.assertEqual(select_indices(dataset, 4), [0, 9, 10, 12])
        self.assertEqual(len(select_indices(dataset, 1)), 1)
        self.assertEqual(select_indices(dataset, 0), list(range(13)))
        self.assertEqual(select_indices(dataset, 100), list(range(13)))

    def test_evaluation_unnormalizes_masks_and_saves_predictions(self):
        class Dataset:
            meta = types.SimpleNamespace(has_language_columns=False, camera_keys=[])

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return {
                    "action": torch.zeros((2, 2)),
                    "action_is_pad": torch.tensor([False, index == 0]),
                    "prediction": torch.tensor([[.1, .3], [100., 100.]]) if index == 0 else torch.tensor([[.2, .4], [0., .6]]),
                    "loss": torch.tensor(2. if index == 0 else 4.),
                    "episode_index": torch.tensor(index),
                    "frame_index": torch.tensor(5),
                }

        class Policy:
            calls = 0

            def eval(self):
                pass

            def reset(self):
                pass

            def forward(self, batch):
                return batch["loss"].mean(), {}

            def predict_action_chunk(self, batch):
                if "action" in batch:
                    raise AssertionError("targets leaked into inference")
                self.calls += 1
                return batch["prediction"]

        collate = types.ModuleType("lerobot.utils.collate")
        collate.lerobot_collate_fn = None
        policy = Policy()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch.dict("sys.modules", {"lerobot.utils.collate": collate}), contextlib.redirect_stdout(io.StringIO()):
                result = evaluate(policy, lambda batch: batch, lambda actions: actions * 10,
                                  Dataset(), [0, 1], folder, torch.device("cpu"), False)
            self.assertAlmostEqual(result["mae"], 16 / 6, places=5)
            self.assertAlmostEqual(result["rmse"], np.sqrt(11), places=5)
            self.assertAlmostEqual(result["flow_matching_loss"], 10 / 3)
            self.assertEqual(policy.calls, 3)  # one warmup + two measured chunks
            self.assertEqual(result["samples"], 2)
            self.assertIsNone(result["vram_peak_allocated_mib"])
            self.assertGreater(result["ram_rss_sampled_peak_mib"], 0)
            self.assertGreater(result["chunk_latency_mean_ms"], 0)
            files = sorted((folder / "inference").glob("*.npz"))
            self.assertEqual(len(files), 2)
            with np.load(files[0], allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["predicted_actions"][0, 0], [1, 3])
                np.testing.assert_array_equal(saved["valid_action_mask"], [[True, False]])
                self.assertEqual(saved["episode_index"].item(), 0)
                self.assertIn("latency_ms", saved)


if __name__ == "__main__":
    unittest.main()
