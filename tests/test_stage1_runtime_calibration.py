"""Stage 1 mapping, approval/geometry, review gates, and real CPU execution."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

from src.stage1_calibration.zone_calibrator import (
    CALIBRATION_CLASSES, auto_detect_zones, draw_calibration, geometry_checks,
    save_approved_zones, validate_room_zones, validate_runtime_calibration,
)
from src.stage1_calibration.dataset_sanitizer import canonical_name, detection_rows, label_text
from src.stage1_calibration.annotation_review import build_clean_evaluation, reviewed_items
from src.stage1_calibration.validate_stage1 import evaluate_reviewed
from src.stage2_monitors.common import RoomZones


FRAME = np.zeros((480, 640, 3), np.uint8)
NAMES = {56: "chair", 57: "couch", 59: "bed", 60: "dining table", 62: "tv", 63: "laptop", 69: "oven"}


def detector(rows):
    model = MagicMock()
    model.names = NAMES
    result = MagicMock()
    result.boxes.data = torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)
    model.predict.return_value = [result]
    return model


def proposed():
    return auto_detect_zones(FRAME, detector([[10, 80, 200, 220, .9, 59]]), source="synthetic_test")


class CalibrationUnitTests(unittest.TestCase):
    def test_filtering_mapping_and_context_never_fabricate_desk_or_stove(self):
        model = detector([[20, 100, 150, 300, .9, 56], [160, 100, 260, 300, .8, 57],
                          [280, 100, 450, 300, .8, 59], [200, 200, 450, 400, .9, 60],
                          [230, 140, 300, 220, .9, 63], [480, 200, 600, 350, .9, 69]])
        config = auto_detect_zones(FRAME, model)
        kwargs = model.predict.call_args.kwargs
        self.assertEqual(kwargs["classes"], [56, 57, 59, 60, 62, 63, 69])
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["imgsz"], 480)
        types = {z["type"] for z in config["zones"].values()}
        self.assertEqual(types, {"chair", "couch", "bed", "dining_table", "oven"})
        self.assertNotIn("desk", types)
        self.assertNotIn("stove", types)
        self.assertTrue(all(z["approved"] is False for z in config["zones"].values()))
        self.assertEqual(config["context_objects"][0]["source_class_id"], 63)
        geometry_checks(config, FRAME.shape)

    def test_wrong_model_taxonomy_is_rejected(self):
        model = detector([])
        model.names = {0: "Bed"}
        with self.assertRaisesRegex(ValueError, "COCO"):
            auto_detect_zones(FRAME, model)

    def test_empty_detection_has_no_invented_defaults(self):
        config = auto_detect_zones(FRAME, detector([]))
        self.assertEqual(config["zones"], {})
        self.assertEqual(config["calibration"]["status"], "proposed")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "confirmation"):
                save_approved_zones(config, Path(directory) / "zones.json", reviewer="synthetic-test")

    def test_approval_round_trip_and_approved_file_preservation(self):
        config = proposed()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.json"
            with self.assertRaisesRegex(ValueError, "confirmation"):
                save_approved_zones(config, path, reviewer="synthetic-test")
            self.assertFalse(path.exists())
            config["zones"]["bed_0"]["approved"] = True
            value = save_approved_zones(config, path, reviewer="synthetic-test")
            loaded = validate_room_zones(path)
            self.assertEqual(loaded, value)
            validate_runtime_calibration(loaded, FRAME.shape)
            self.assertIsNotNone(RoomZones(loaded["zones"]).get("bed"))
            self.assertTrue(draw_calibration(FRAME, loaded).any())
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_approved_zones(config, path, reviewer="synthetic-test")
            self.assertEqual(path.read_bytes(), original)

    def test_source_resolution_and_unapproved_configuration_fail_closed(self):
        config = proposed()
        with self.assertRaisesRegex(ValueError, "manual confirmation"):
            validate_runtime_calibration(config, FRAME.shape)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_runtime_calibration(config, (720, 1280, 3))

    def test_geometry_and_explicit_overlap_acknowledgment(self):
        config = proposed()
        config["zones"]["bed_0"]["approved"] = True
        config["zones"]["chair_0"] = {"type": "chair", "polygon": [[20, 100], [50, 100], [50, 150], [20, 150]], "approved": True}
        self.assertEqual(len(geometry_checks(config, FRAME.shape)), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.json"
            with self.assertRaisesRegex(ValueError, "overlapping"):
                save_approved_zones(config, path, reviewer="synthetic-test")
            saved = save_approved_zones(config, path, reviewer="synthetic-test", overlap_reviewed=True)
            validate_runtime_calibration(saved, FRAME.shape)
        for points in ([[0, 0], [1, 1]], [[0, 0], [100, 100], [0, 100], [100, 0]],
                       [[-1, 0], [10, 0], [10, 10]], [[0, 0], [1000, 0], [10, 10]]):
            with self.subTest(points=points):
                bad = proposed()
                bad["zones"]["bed_0"]["polygon"] = points
                with self.assertRaises(ValueError):
                    geometry_checks(bad, FRAME.shape)

    def test_partial_polygon_is_renderable_during_manual_creation(self):
        config = proposed()
        config["zones"]["bed_0"]["polygon"] = []
        self.assertEqual(draw_calibration(FRAME, config).shape, FRAME.shape)


class AnnotationReviewTests(unittest.TestCase):
    def make_pack(self, root):
        image = root / "source.jpg"
        cv2.imwrite(str(image), np.zeros((40, 40, 3), np.uint8))
        from src.stage1_calibration.dataset_sanitizer import digest
        review = root / "review"
        (review / "decisions").mkdir(parents=True)
        item = {"id": "000", "image": str(image), "image_sha256": digest(image),
                "sanitized_original_boxes": [[0, .5, .5, .5, .5]]}
        manifest = {"items": [item], "class_names": ["bed", "dining_table", "table"],
                    "archive_sha256": "synthetic_fixture"}
        (review / "manifest.json").write_text(json.dumps(manifest))
        decision = {"id": "000", "image_sha256": item["image_sha256"], "status": "pending",
                    "boxes": item["sanitized_original_boxes"], "reviewer": "", "reviewed_at": ""}
        path = review / "decisions/000.json"
        path.write_text(json.dumps(decision))
        return review, path, decision

    def test_mixed_rows_and_aliases_preserve_class_semantics(self):
        rows = detection_rows("0 .5 .5 .2 .4\n1 .1 .2 .3 .2 .3 .4", 25)
        self.assertTrue(np.allclose(rows[0], [0, .5, .5, .2, .4]))
        self.assertTrue(np.allclose(rows[1], [1, .2, .3, .2, .2]))
        self.assertTrue(all(len(line.split()) == 5 for line in label_text(rows).splitlines()))
        self.assertEqual(canonical_name("Sofa"), "couch")
        self.assertNotEqual(canonical_name("Dining Table"), canonical_name("Table"))
        self.assertNotEqual(canonical_name("Dining Table"), canonical_name("Desk"))
        for text in ("0 .5 .5 -.1 .2", "0 .5 .5 2 .2", "99 .5 .5 .2 .2", "0 nan .5 .2 .2"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                detection_rows(text, 25)

    def test_pending_annotations_never_create_ground_truth_or_clean_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, _, _ = self.make_pack(root)
            clean, output = root / "clean", root / "report"
            result = evaluate_reviewed(review, clean, output)
            self.assertEqual(result["status"], "unavailable")
            self.assertFalse(clean.exists())
            self.assertEqual(len((output / "clean_metrics.csv").read_text().splitlines()), 1)

    def test_only_explicit_reviews_enter_clean_copy_and_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, path, decision = self.make_pack(root)
            source = root / "source.jpg"
            before = source.read_bytes()
            decision.update(status="accepted", reviewer="synthetic-test", reviewed_at=datetime.now(timezone.utc).isoformat())
            path.write_text(json.dumps(decision))
            clean = root / "clean"
            built = build_clean_evaluation(review, clean)
            self.assertEqual(built["reviewed_images"], 1)
            self.assertTrue((clean / "test/images/source.jpg").is_symlink())
            self.assertEqual(source.read_bytes(), before)
            config = json.loads((clean / "data.yaml").read_text())
            self.assertEqual(config["names"], ["bed", "dining_table", "table"])

    def test_edited_or_fractional_class_ids_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            review, path, decision = self.make_pack(Path(directory))
            decision.update(status="accepted", reviewer="synthetic-test", reviewed_at=datetime.now(timezone.utc).isoformat(),
                            boxes=[[0, .4, .4, .2, .2]])
            path.write_text(json.dumps(decision))
            with self.assertRaisesRegex(ValueError, "corrected"):
                reviewed_items(review)
            decision.update(status="corrected", boxes=[[1.5, .5, .5, .2, .2]])
            path.write_text(json.dumps(decision))
            with self.assertRaises(ValueError):
                reviewed_items(review)

    def test_synthetic_reviewed_metrics_account_for_false_positives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, path, decision = self.make_pack(root)
            decision.update(status="accepted", reviewer="synthetic-test", reviewed_at=datetime.now(timezone.utc).isoformat())
            path.write_text(json.dumps(decision))
            model = detector([[10, 10, 30, 30, .9, 59], [10, 10, 30, 30, .8, 59]])
            model.names = {59: "bed"}
            with patch("src.stage1_calibration.validate_stage1.load_cpu_model", return_value=model):
                result = evaluate_reviewed(review, root / "clean", root / "output")
            self.assertEqual(result["images"], 1)
            import csv
            with (root / "output/clean_metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["tp"], "1")
            self.assertEqual(rows[0]["fp"], "1")
            self.assertEqual(rows[0]["missed"], "0")
            self.assertEqual(float(rows[0]["precision"]), .5)


class CPUCalibrationIntegrationTests(unittest.TestCase):
    def test_demo_geometry_cannot_enable_live_camera_monitoring(self):
        from contextlib import redirect_stderr
        import io
        from src.main_pipeline import main

        for source in ("0", "rtsp://example.local/live"):
            with self.subTest(source=source), redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit):
                    main(["--source", source, "--headless", "--allow-demo-zones"])
                self.assertIn("forbidden for live cameras", stderr.getvalue())

    def test_existing_cpu_export_on_a_real_indoor_frame(self):
        from src.stage1_calibration.dataset_sanitizer import CACHE
        from src.utils.export_cpu_models import load_cpu_model
        images = sorted((CACHE / "test/images").glob("ikea-wake-up*"))
        self.assertTrue(images, "Generate the sanitized source cache before running integration checks")
        frame = cv2.imread(str(images[0]))
        config = auto_detect_zones(frame, load_cpu_model("yolo11n.pt"))
        self.assertTrue(config["zones"])
        geometry_checks(config, frame.shape)
        self.assertIn("bed", {z["type"] for z in config["zones"].values()})
        self.assertTrue(all(z["approved"] is False for z in config["zones"].values()))

    def test_camera_startup_calibrates_once_outside_the_shared_scheduler(self):
        from src.main_pipeline import InferenceResult, main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = proposed()
            config["zones"]["bed_0"]["approved"] = True
            path = root / "zones.json"
            save_approved_zones(config, path, reviewer="synthetic-test")
            models = MagicMock()

            def infer(frame, captured_at, run_pose, run_objects):
                return InferenceResult(captured_at, () if run_pose else None,
                                       () if run_objects else None, .1 if run_pose else None,
                                       .1 if run_objects else None)

            models.infer.side_effect = infer
            capture = MagicMock()
            capture.isOpened.return_value = True
            capture.get.return_value = 30
            capture.read.side_effect = [(True, FRAME.copy()), (True, FRAME.copy()), (False, None)]
            with patch("src.main_pipeline.CPUModels", return_value=models), \
                 patch("src.main_pipeline.cv2.VideoCapture", return_value=capture), \
                 patch("src.stage1_calibration.zone_calibrator.auto_detect_zones", return_value=proposed()) as auto:
                main(["--source", "0", "--zones", str(path), "--headless", "--event-log", str(root / "events.jsonl")])
            self.assertEqual(auto.call_count, 1)
            self.assertTrue(capture.release.called)


if __name__ == "__main__":
    unittest.main()
