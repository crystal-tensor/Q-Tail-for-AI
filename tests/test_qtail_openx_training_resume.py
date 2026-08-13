import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.qtail_train_openx_cached import resumable_train_head


class OpenXTrainingResumeTest(unittest.TestCase):
    def test_interrupted_resume_matches_uninterrupted_training(self) -> None:
        rng = np.random.default_rng(11)
        features = rng.normal(size=(64, 10)).astype(np.float32)
        target = rng.exponential(size=64).astype(np.float32)
        target /= target.sum()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_history, baseline_prediction, baseline_model = resumable_train_head(
                features,
                target,
                80,
                11,
                phase="qtail",
                resume_dir=root / "baseline",
                progress_path=root / "baseline-progress.json",
            )
            with self.assertRaisesRegex(RuntimeError, "controlled interruption"):
                resumable_train_head(
                    features,
                    target,
                    80,
                    11,
                    phase="qtail",
                    resume_dir=root / "resume",
                    progress_path=root / "resume-progress.json",
                    stop_after_step=37,
                )
            resumed_history, resumed_prediction, resumed_model = resumable_train_head(
                features,
                target,
                80,
                11,
                phase="qtail",
                resume_dir=root / "resume",
                progress_path=root / "resume-progress.json",
            )
            self.assertEqual(baseline_history, resumed_history)
            np.testing.assert_array_equal(baseline_prediction, resumed_prediction)
            for name, baseline_parameter in baseline_model.state_dict().items():
                self.assertTrue(
                    torch.equal(
                        baseline_parameter, resumed_model.state_dict()[name]
                    ),
                    name,
                )

    def test_changed_input_rejects_resume_checkpoint(self) -> None:
        rng = np.random.default_rng(17)
        features = rng.normal(size=(32, 10)).astype(np.float32)
        target = np.full(32, 1 / 32, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "controlled interruption"):
                resumable_train_head(
                    features,
                    target,
                    20,
                    11,
                    phase="source",
                    resume_dir=root,
                    progress_path=root / "progress.json",
                    stop_after_step=7,
                )
            changed_features = features.copy()
            changed_features[0, 0] += 1
            resumable_train_head(
                changed_features,
                target,
                20,
                11,
                phase="source",
                resume_dir=root,
                progress_path=root / "progress.json",
            )
            progress = __import__("json").loads(
                (root / "progress.json").read_text(encoding="utf-8")
            )
            self.assertFalse(progress["resumed"])


if __name__ == "__main__":
    unittest.main()
