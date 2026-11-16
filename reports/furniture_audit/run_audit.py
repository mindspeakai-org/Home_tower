"""Offline, read-only dataset/model audit. Outputs stay beside this script.

Run from the project root with .venv/bin/python reports/furniture_audit/run_audit.py.
No dataset downloads, training, runtime edits, or original label rewrites.
"""

import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import time
import zipfile

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ARCHIVE = next((ROOT / "data").glob("*.zip"))
WEIGHT = ROOT / "weights" / "yolo11n.pt"
EXPORTED = ROOT / "weights" / "yolo11n_openvino_model"
COCO_TO_ORIGINAL = {59: 0, 56: 4, 57: 18, 60: 8}
CONF = 0.25
LOW_CONF = 0.001


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def parse_label(text, context):
    """Decode each YOLO bbox or segment independently, preserving original IDs."""
    labels, formats, issues = [], [], []
    for index, line in enumerate(text.strip().splitlines()):
        values = np.array([float(x) for x in line.split()], dtype=float)
        if (not np.isfinite(values).all() or len(values) < 5 or values[0] != int(values[0])
                or not 0 <= values[0] < 25):
            raise ValueError(f"Invalid label: {context}:{index + 1}: {line}")
        if len(values) == 5:
            _, x, y, w, h = values
            xyxy = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            kind = "bbox"
        elif len(values) >= 7 and len(values) % 2 == 1:
            points = values[1:].reshape(-1, 2)
            xyxy = np.r_[points.min(axis=0), points.max(axis=0)]
            kind = "segment"
        else:
            raise ValueError(f"Unsupported label shape: {context}:{index + 1}")
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            raise ValueError(f"Degenerate ground truth: {context}:{index + 1}")
        if xyxy.min() < -1e-5 or xyxy.max() > 1 + 1e-5:
            issues.append({"file": context, "line": index + 1, "box": xyxy.tolist()})
        labels.append([int(values[0]), *xyxy.tolist()])
        formats.append(kind)
    return np.array(labels, dtype=float).reshape(-1, 5), formats, issues


def inventory(z):
    raw_yaml = z.read("data.yaml").decode()
    names = ast.literal_eval(next(line.split(": ", 1)[1] for line in raw_yaml.splitlines()
                                  if line.startswith("names:")))
    report = {"archive": str(ARCHIVE.relative_to(ROOT)), "archive_sha256": sha256(ARCHIVE.read_bytes()),
              "classes": dict(enumerate(names)), "splits": {}, "out_of_bounds_boxes": [],
              "mixed_format_files": [], "label_format_counts": {}, "decoding_errors": []}
    image_hashes, source_stems = defaultdict(list), defaultdict(list)
    for split in ("train", "valid", "test", "export"):
        images = sorted(n for n in z.namelist() if n.startswith(split + "/images/") and not n.endswith("/"))
        labels = sorted(n for n in z.namelist() if n.startswith(split + "/labels/") and n.endswith(".txt"))
        counts, formats = Counter(), Counter()
        empty = 0
        for path in labels:
            boxes, kinds, issues = parse_label(z.read(path).decode(), path)
            counts.update(boxes[:, 0].astype(int).tolist())
            formats.update(kinds)
            empty += len(boxes) == 0
            report["out_of_bounds_boxes"].extend(issues)
            if len(set(kinds)) > 1:
                report["mixed_format_files"].append(path)
        missing = []
        for path in images:
            data = z.read(path)
            image_hashes[sha256(data)].append(path)
            source_stems[re.sub(r"\.rf\.[^.]+$", "", Path(path).stem)].append(path)
            label_path = str(Path(path.replace("/images/", "/labels/")).with_suffix(".txt"))
            if label_path not in labels:
                missing.append(path)
        report["splits"][split] = {
            "images": len(images), "label_files": len(labels), "empty_labels": int(empty),
            "objects": sum(counts.values()), "per_class": dict(counts), "missing_labels": missing,
        }
        report["label_format_counts"][split] = dict(formats)
    exact_cross = [paths for paths in image_hashes.values()
                   if len({p.split('/')[0] for p in paths}) > 1]
    stem_cross = [paths for paths in source_stems.values()
                  if len({p.split('/')[0] for p in paths}) > 1]
    report["duplicates"] = {
        "exact_duplicate_groups": [paths for paths in image_hashes.values() if len(paths) > 1],
        "exact_cross_split_groups": exact_cross,
        "source_stem_cross_split_candidates": stem_cross,
        "source_stem_warning": "Shared basename is only a leakage candidate, not proof of image identity",
    }
    dump("inventory.json", report)
    for name in ("data.yaml", "README.dataset.txt", "README.roboflow.txt"):
        path = OUT / "original_metadata" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(z.read(name))
    return names, report


def mapped_predictions(predictions, grouped=False):
    converted = []
    for row in predictions:
        coco_id = int(row[5])
        if coco_id in COCO_TO_ORIGINAL:
            converted.append([*row[:5], COCO_TO_ORIGINAL[coco_id]])
    return np.array(converted, dtype=float).reshape(-1, 6)


def matches(validator, gt, pred):
    if not len(gt) or not len(pred):
        return np.zeros((len(pred), 10), dtype=bool)
    iou = box_iou(torch.tensor(gt[:, 1:], dtype=torch.float32),
                  torch.tensor(pred[:, :4], dtype=torch.float32))
    return validator.match_predictions(torch.tensor(pred[:, 5]), torch.tensor(gt[:, 0]), iou).numpy()


def evaluate(records, names, validator, grouped=False):
    all_tp, all_conf, all_pred, all_gt = [], [], [], []
    counts = {i: Counter() for i in range(25)}
    per_image = []
    for record in records:
        gt = np.asarray(record["ground_truth"], dtype=float).reshape(-1, 5).copy()
        if grouped:
            gt[gt[:, 0] == 20, 0] = 8
        pred = mapped_predictions(record["predictions"])
        tp = matches(validator, gt, pred)
        all_tp.append(tp)
        all_conf.extend(pred[:, 4].tolist())
        all_pred.extend(pred[:, 5].tolist())
        all_gt.extend(gt[:, 0].tolist())
        keep = pred[:, 4] >= CONF
        image_counts = {}
        for identifier in range(25):
            selected = keep & (pred[:, 5] == identifier)
            target_count = int((gt[:, 0] == identifier).sum())
            correct = int(tp[selected, 0].sum())
            predicted = int(selected.sum())
            counts[identifier].update(gt=target_count, tp=correct, fp=predicted - correct,
                                      fn=target_count - correct, predictions=predicted)
            if target_count or predicted:
                image_counts[identifier] = dict(gt=target_count, tp=correct,
                                                fp=predicted - correct, fn=target_count - correct)
        per_image.append({"image": record["image"], "classes": image_counts})
    metrics = ap_per_class(np.concatenate(all_tp), np.array(all_conf), np.array(all_pred),
                           np.array(all_gt), names=dict(enumerate(names)), plot=False)
    ap_values = {int(identifier): (float(ap[0]), float(ap.mean()))
                 for identifier, ap in zip(metrics[6], metrics[5])}
    rows = []
    for identifier, name in enumerate(names):
        if grouped and identifier == 20:
            continue
        c = counts[identifier]
        ap50, ap5095 = ap_values.get(identifier, (None, None))
        rows.append({"id": identifier, "name": "table (Dining Table + Table)" if grouped and identifier == 8 else name,
                     "model_coco_id": next((k for k, v in COCO_TO_ORIGINAL.items() if v == identifier), None),
                     "gt": c["gt"], "predictions": c["predictions"], "tp": c["tp"], "fp": c["fp"],
                     "missed": c["fn"], "precision": c["tp"] / c["predictions"] if c["predictions"] else None,
                     "recall": c["tp"] / c["gt"] if c["gt"] else None,
                     "ap50": ap50, "ap50_95": ap5095})
    stem = "grouped_table" if grouped else "strict_original"
    dump(f"{stem}_metrics.json", rows)
    with (OUT / f"{stem}_metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(f"{stem}_per_image.json", per_image)
    return rows


def self_checks(validator):
    gt = np.array([[0, 0, 0, 100, 100]], dtype=float)
    pred = np.array([[0, 0, 100, 100, .9, 0], [0, 0, 100, 100, .8, 0]], dtype=float)
    assert matches(validator, gt, pred)[:, 0].tolist() == [True, False]
    pred[:, 5] = 4
    assert not matches(validator, gt, pred).any()
    labels, kinds, _ = parse_label("0 0.5 0.5 0.2 0.4\n4 0.1 0.2 0.3 0.2 0.3 0.4", "self_check")
    assert kinds == ["bbox", "segment"] and np.allclose(labels[1, 1:], [.1, .2, .3, .4])
    empty = np.empty((0, 6))
    assert matches(validator, gt, empty).shape == (0, 10)
    dump("self_checks.json", {"confidence_ordered_duplicate_matching": "pass",
                              "wrong_class_rejected": "pass", "mixed_row_formats": "pass",
                              "empty_predictions": "pass"})


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        names, info = inventory(z)
        print("INVENTORY", {k: (v['images'], v['objects']) for k, v in info['splits'].items()}, flush=True)
        checkpoint = YOLO(str(WEIGHT))
        ckpt = checkpoint.ckpt
        metadata = {key: ckpt.get(key) for key in ("date", "version", "epoch", "license", "train_args", "train_metrics")}
        metadata.update(sha256=sha256(WEIGHT.read_bytes()), checkpoint=str(WEIGHT.relative_to(ROOT)),
                        classes=checkpoint.names, task=checkpoint.task,
                        embedded_train_results_epochs=len(ckpt.get("train_results", {}).get("epoch", [])))
        dump("checkpoint_evidence.json", metadata)
        validator = BaseValidator(save_dir=OUT / "native_metric_engine", args={"device": "cpu", "plots": False})
        validator.iouv = torch.linspace(0.5, 0.95, 10)
        self_checks(validator)
        model = YOLO(str(EXPORTED), task="detect")
        warm = np.zeros((480, 480, 3), np.uint8)
        model.predict(warm, device="cpu", imgsz=480, rect=False, conf=LOW_CONF, iou=0.7, verbose=False)
        assert model.names == checkpoint.names
        test_images = sorted(n for n in z.namelist() if n.startswith("test/images/") and not n.endswith("/"))
        records = []
        prediction_path = OUT / "test_predictions.jsonl"
        image_sizes = Counter()
        with prediction_path.open("w") as stream:
            for index, name in enumerate(test_images):
                frame = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Unreadable held-out image: {name}")
                height, width = frame.shape[:2]
                image_sizes[f"{width}x{height}"] += 1
                label_name = str(Path(name.replace("/images/", "/labels/")).with_suffix(".txt"))
                gt, _, _ = parse_label(z.read(label_name).decode(), label_name)
                gt[:, 1:] *= [width, height, width, height]
                tick = time.perf_counter()
                result = model.predict(frame, device="cpu", imgsz=480, rect=False,
                                       conf=LOW_CONF, iou=0.7, max_det=300, verbose=False)[0]
                record = {"image": name, "shape": [height, width], "ground_truth": gt.tolist(),
                          "predictions": result.boxes.data.cpu().numpy().tolist(),
                          "latency_ms": (time.perf_counter() - tick) * 1000}
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if (index + 1) % 50 == 0 or index + 1 == len(test_images):
                    print(f"CPU EVALUATION {index + 1}/{len(test_images)}", flush=True)
        strict = evaluate(records, names, validator)
        grouped = evaluate(records, names, validator, grouped=True)
        distribution = Counter(int(row[5]) for record in records for row in record['predictions'] if row[4] >= CONF)
        summary = {"images": len(records), "imgsz": 480, "device": "cpu", "backend": "OpenVINO",
                   "operating_confidence": CONF, "ap_confidence_floor": LOW_CONF, "nms_iou": 0.7,
                   "matching_iou": 0.5, "max_det": 300, "ultralytics": ultralytics.__version__,
                   "torch": torch.__version__, "platform": platform.platform(), "image_sizes": dict(image_sizes),
                   "mean_inference_call_ms": float(np.mean([r['latency_ms'] for r in records])),
                   "all_coco_prediction_counts_at_conf025": {checkpoint.names[k]: v for k, v in distribution.items()},
                   "strict_map50_present_classes": float(np.mean([r['ap50'] for r in strict if r['ap50'] is not None])),
                   "grouped_map50_present_classes": float(np.mean([r['ap50'] for r in grouped if r['ap50'] is not None])),
                   "elapsed_seconds": time.perf_counter() - started}
        dump("evaluation_summary.json", summary)
        print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
