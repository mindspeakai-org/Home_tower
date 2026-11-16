"""CPU real-image proposal checks and evaluation gated by reviewed annotations."""

import argparse
import csv
import json
from pathlib import Path
import platform
import resource
import shutil
import time

import cv2
import numpy as np
import psutil
import torch
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou

from .annotation_review import REVIEW, build_clean_evaluation
from .dataset_sanitizer import ROOT, canonical_name
from .zone_calibrator import auto_detect_zones, draw_calibration, geometry_checks
from src.utils.export_cpu_models import load_cpu_model


OUTPUT = ROOT / "reports/stage1_validation"
FIELDS = ["id", "class", "gt", "predictions", "tp", "fp", "missed", "precision", "recall", "ap50"]


def check_real_rooms(output: Path = OUTPUT) -> dict:
    examples = json.loads((ROOT / "reports/furniture_audit/indoor_examples.json").read_text())
    selected = [examples[i] for i in (0, 1, 4, 5, 9, 12, 16, 18, 19, 22, 24, 27)]
    cache = ROOT / "datasets/furniture_original_cache"
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    model = load_cpu_model("yolo11n.pt")
    warmup_seconds = time.perf_counter() - started
    rss_after_warmup = process.memory_info().rss
    folder = output / "room_proposals"
    folder.mkdir(parents=True, exist_ok=True)
    checks, panels, timings = [], [], []
    peak_rss = rss_after_warmup
    for index, record in enumerate(selected):
        image_path = cache / record["image"]
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Missing indoor fixture: {image_path}")
        started = time.perf_counter()
        config = auto_detect_zones(frame, model, source=record["image"])
        latency = (time.perf_counter() - started) * 1000
        timings.append(latency)
        peak_rss = max(peak_rss, process.memory_info().rss)
        if not config["zones"]:
            raise AssertionError(f"No supported furniture proposals on selected fixture {record['image']}")
        overlaps = geometry_checks(config, frame.shape)
        path = folder / f"{index:02d}.json"
        path.write_text(json.dumps(config, indent=2))
        reloaded = json.loads(path.read_text())
        if reloaded != config:
            raise AssertionError("Proposal serialization changed values")
        canvas = draw_calibration(frame, reloaded)
        cv2.imwrite(str(folder / f"{index:02d}.png"), canvas)
        checks.append({"image": record["image"], "latency_ms": latency,
                       "proposed_types": sorted({z["type"] for z in config["zones"].values()}),
                       "zone_count": len(config["zones"]), "overlaps": overlaps,
                       "status": "unreviewed_model_proposals", "saved_as_runtime_calibration": False})
        panel = np.full((446, 416, 3), 20, np.uint8)
        panel[:416] = cv2.resize(canvas, (416, 416))
        cv2.putText(panel, f"{index:02d} UNCONFIRMED | {latency:.1f} ms", (5, 437),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        panels.append(panel)
    grid = np.vstack([np.hstack(panels[i:i + 3]) for i in range(0, 12, 3)])
    cv2.imwrite(str(ROOT / "reports/furniture_audit/fixed_calibration_verification.png"), grid)
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    result = {"status": "real_images_real_CPU_predictions_not_manual_approval", "images": checks,
              "device": "cpu", "backend": "existing OpenVINO/ONNX CPU loader", "imgsz": 480,
              "cpu": cpu, "model_load_and_warmup_seconds": warmup_seconds,
              "mean_proposal_latency_ms": float(np.mean(timings)), "p95_proposal_latency_ms": float(np.percentile(timings, 95)),
              "rss_before_bytes": rss_before, "rss_after_warmup_bytes": rss_after_warmup,
              "sampled_peak_rss_bytes": peak_rss, "process_high_water_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "unsupported": ["desk", "stove"], "precision_recall_claim": None}
    (output / "real_image_execution.json").write_text(json.dumps(result, indent=2))
    return result


def evaluate_reviewed(review_dir: Path, clean_dir: Path, output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    status = build_clean_evaluation(review_dir, clean_dir)
    if status["status"] == "unavailable":
        with (output / "clean_metrics.csv").open("w") as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()
        (output / "clean_evaluation_status.json").write_text(json.dumps(status, indent=2))
        return status
    manifest = json.loads((review_dir / "manifest.json").read_text())
    names = manifest["class_names"]
    model = load_cpu_model("yolo11n.pt")
    mapping = {i: names.index(canonical_name(name)) for i, name in model.names.items()
               if canonical_name(name) in names}
    validator = BaseValidator(save_dir=output / "clean_metric_engine", args={"device": "cpu", "plots": False})
    validator.iouv = torch.linspace(.5, .95, 10)
    all_tp, all_conf, all_cls, all_gt, records = [], [], [], [], []
    for entry in status["items"]:
        frame = cv2.imread(entry["image"])
        height, width = frame.shape[:2]
        gt = []
        for cls, x, y, w, h in entry["decision"]["boxes"]:
            gt.append([cls, (x - w / 2) * width, (y - h / 2) * height,
                       (x + w / 2) * width, (y + h / 2) * height])
        gt = np.array(gt).reshape(-1, 5)
        result = model.predict(frame, device="cpu", imgsz=480, rect=False, conf=.001,
                               iou=.7, max_det=300, verbose=False)[0]
        pred = np.array([[*row[:5], mapping[int(row[5])]] for row in result.boxes.data.cpu().numpy()
                         if int(row[5]) in mapping]).reshape(-1, 6)
        iou = box_iou(torch.tensor(gt[:, 1:], dtype=torch.float32), torch.tensor(pred[:, :4], dtype=torch.float32))
        tp = (validator.match_predictions(torch.tensor(pred[:, 5]), torch.tensor(gt[:, 0]), iou).numpy()
              if len(pred) and len(gt) else np.zeros((len(pred), 10), bool))
        all_tp.append(tp)
        all_conf.extend(pred[:, 4])
        all_cls.extend(pred[:, 5])
        all_gt.extend(gt[:, 0])
        records.append((entry, frame, gt, pred, tp, iou.numpy()))
    tp = np.concatenate(all_tp)
    conf, classes, target_classes = map(np.asarray, (all_conf, all_cls, all_gt))
    aps = ap_per_class(tp, conf, classes, target_classes, plot=False)
    ap50 = dict(zip(aps[6].astype(int), aps[5][:, 0]))
    rows = []
    for identifier, name in enumerate(names):
        selected = (classes == identifier) & (conf >= .25)
        truth = int((target_classes == identifier).sum())
        correct = int(tp[selected, 0].sum())
        predicted = int(selected.sum())
        rows.append(dict(id=identifier, **{"class": name}, gt=truth, predictions=predicted,
                         tp=correct, fp=predicted - correct, missed=truth - correct,
                         precision=correct / predicted if predicted else None,
                         recall=correct / truth if truth else None, ap50=float(ap50[identifier]) if identifier in ap50 else None))
    with (output / "clean_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    cases = output / "failure_cases"
    cases.mkdir(exist_ok=True)
    for entry, frame, gt, pred, correct, iou in records:
        matched = set()
        for i, p in enumerate(pred):
            if p[4] < .25:
                continue
            if correct[i, 0]:
                eligible = [j for j, g in enumerate(gt) if j not in matched and int(g[0]) == int(p[5])]
                if eligible:
                    matched.add(max(eligible, key=lambda j: iou[j, i]))
            color = (0, 220, 0) if correct[i, 0] else (0, 165, 255)
            cv2.rectangle(frame, tuple(p[:2].astype(int)), tuple(p[2:4].astype(int)), color, 2)
        for i, g in enumerate(gt):
            if i not in matched:
                cv2.rectangle(frame, tuple(g[1:3].astype(int)), tuple(g[3:5].astype(int)), (0, 0, 255), 2)
        cv2.imwrite(str(cases / f"reviewed_{entry['id']}.png"), frame)
    summary = {"status": "measured_on_manually_reviewed_subset", "images": len(records),
               "conf": .25, "iou": .5, "imgsz": 480, "device": "cpu",
               "warning": "Unsupported COCO classes retain zero recall; dining_table and table remain distinct"}
    (output / "clean_evaluation_status.json").write_text(json.dumps(summary, indent=2))
    return summary


def export_unreviewed_baseline_cases(output: Path = OUTPUT):
    cases = output / "failure_cases"
    cases.mkdir(parents=True, exist_ok=True)
    descriptions = {2: "missed_bed", 12: "unlabelled_chairs_apparent_FP", 17: "table_confused_as_bed",
                    21: "spurious_oven", 22: "empty_annotations_visible_workstation", 24: "correct_bed"}
    for index, name in descriptions.items():
        shutil.copyfile(ROOT / f"reports/furniture_audit/indoor_examples/{index:02d}.jpg",
                        cases / f"unreviewed_baseline_{name}.jpg")
    (cases / "STATUS.json").write_text(json.dumps({
        "status": "unreviewed_original_baseline_only",
        "reason": "Reviewed clean false positives/misses cannot be labelled until manual review",
        "legend": "Original annotations green; stock model orange; confidence >=0.25",
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    parser.add_argument("--clean-dir", type=Path, default=ROOT / "datasets/furniture_reviewed_evaluation")
    parser.add_argument("--skip-rooms", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    export_unreviewed_baseline_cases()
    if not args.skip_rooms:
        check_real_rooms()
    print(json.dumps(evaluate_reviewed(args.review_dir, args.clean_dir), indent=2))


if __name__ == "__main__":
    main()
