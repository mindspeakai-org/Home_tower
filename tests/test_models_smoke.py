"""Load both local checkpoints and run one inference on each available device."""

import unittest

import numpy as np
import torch
from ultralytics import YOLO

from src.stage1_calibration.download_stage1_data import MODEL_NAMES, WEIGHTS_DIR


class ModelSmokeTests(unittest.TestCase):
    def check_models(self, device: str) -> None:
        for name in MODEL_NAMES:
            with self.subTest(model=name, device=device):
                path = WEIGHTS_DIR / name
                self.assertTrue(path.is_file(), f"Download weights first: {path}")
                model = YOLO(str(path)).to(device)
                expected_task = "pose" if "pose" in name else "detect"
                self.assertEqual(model.task, expected_task)
                results = model.predict(
                    np.zeros((320, 320, 3), dtype=np.uint8),
                    device=device,
                    imgsz=320,
                    verbose=False,
                )
                self.assertEqual(len(results), 1)
                self.assertIsNotNone(results[0].boxes)
                if expected_task == "pose":
                    self.assertIsNotNone(results[0].keypoints)

    def test_cpu(self) -> None:
        self.check_models("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU is unavailable")
    def test_cuda(self) -> None:
        self.check_models("cuda:0")


if __name__ == "__main__":
    unittest.main()
