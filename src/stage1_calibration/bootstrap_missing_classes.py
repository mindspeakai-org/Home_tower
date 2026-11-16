"""Export unreviewed workstation/kitchen candidates, never prediction ground truth."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import box

from .dataset_sanitizer import CACHE, ROOT, digest


BOOTSTRAP = ROOT / "datasets/stage1_workstation_kitchen_bootstrap"
PREVIEW = ROOT / "reports/furniture_audit/bootstrapped_desk_stove_preview.png"


def candidate_context(predictions: list, shape: tuple) -> list[dict]:
    """Context only ranks proposals; COCO cannot establish desk/stove truth."""
    detections = [p for p in predictions if p[4] >= 0.25]
    candidates = []
    for row in detections:
        cls = int(row[5])
        target = box(*row[:4])
        if cls == 60:
            context = [int(other[5]) for other in detections if int(other[5]) in {56, 62, 63, 66}
                       and (target.intersects(box(*other[:4])) or (
                           int(other[5]) in {62, 63, 66} and 0 <= row[1] - other[3] <= 40
                           and min(row[2], other[2]) > max(row[0], other[0]))
                       or (int(other[5]) == 56 and target.distance(box(*other[:4])) < 60))]
            if context:
                candidates.append({"candidate": "workstation_table", "source_label": "dining_table",
                                   "source_class_id": 60, "bbox": row[:4], "confidence": row[4],
                                   "context_class_ids": context, "approved": False,
                                   "instruction": "Verify use and draw desk polygon manually; table is not desk"})
        elif cls == 69:
            context = [int(other[5]) for other in detections if int(other[5]) in {68, 71, 72}
                       and target.distance(box(*other[:4])) <= max(shape) * 0.4]
            candidates.append({"candidate": "kitchen_oven", "source_label": "oven",
                               "source_class_id": 69, "bbox": row[:4], "confidence": row[4],
                               "context_class_ids": context, "approved": False,
                               "instruction": "Verify kitchen context and draw cooktop polygon; oven is not stove"})
    return candidates


def bootstrap_candidates(output: Path = BOOTSTRAP, preview: Path = PREVIEW,
                         local_images: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError("Candidate exports are immutable; choose a new output version")
    audit = ROOT / "reports/furniture_audit"
    evidence = json.loads((audit / "checkpoint_evidence.json").read_text())
    if evidence["sha256"] != digest(ROOT / "weights/yolo11n.pt"):
        raise ValueError("Stock checkpoint changed; cached predictions cannot be reused")
    source = json.loads((CACHE / "source_manifest.json").read_text())
    inventory = json.loads((audit / "inventory.json").read_text())
    if source["archive_sha256"] != inventory["archive_sha256"]:
        raise ValueError("Cached predictions refer to a different archive")
    records = []
    for line in (audit / "test_predictions.jsonl").read_text().splitlines():
        record = json.loads(line)
        records.append({**record, "image_path": str(CACHE / record["image"]),
                        "prediction_origin": "existing CPU audit predictions, not re-audited"})
    if local_images is not None:
        from src.utils.export_cpu_models import load_cpu_model
        model = load_cpu_model("yolo11n.pt")
        for path in sorted(local_images.iterdir()):
            if path.suffix.casefold() not in {".jpg", ".jpeg", ".png"}:
                continue
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Unreadable local image: {path}")
            result = model.predict(frame, device="cpu", imgsz=480, rect=False, verbose=False)[0]
            records.append({"image_path": str(path.resolve()), "image": str(path),
                            "shape": list(frame.shape[:2]), "predictions": result.boxes.data.cpu().numpy().tolist(),
                            "prediction_origin": "new local CPU inference"})
    (output / "images").mkdir(parents=True)
    (output / "proposals").mkdir()
    entries, panels = [], []
    for record in records:
        candidates = candidate_context(record["predictions"], tuple(record["shape"]))
        # Monitor+keyboard without a COCO table is useful for manual desk drawing,
        # but is not a desk detection and supplies no automatic desk rectangle.
        context = {int(p[5]) for p in record["predictions"] if p[4] >= 0.25}
        if {62, 66} <= context and not any(c["candidate"] == "workstation_table" for c in candidates):
            candidates.append({"candidate": "workstation_context_only", "source_label": "tv+keyboard",
                               "bbox": None, "approved": False,
                               "instruction": "No table detected; draw desk polygon manually"})
        if not candidates:
            continue
        image = Path(record["image_path"])
        key = digest(image)[:16]
        link = output / "images" / (key + image.suffix)
        if not link.exists():
            link.symlink_to(image.resolve())
        entry = {"id": key, "image": str(image.resolve()), "image_sha256": digest(image),
                 "source": record["image"], "status": "pending_manual_review",
                 "prediction_origin": record["prediction_origin"], "proposals": candidates}
        (output / "proposals" / (key + ".json")).write_text(json.dumps(entry, indent=2))
        entries.append(entry)
        frame = cv2.imread(str(image))
        for candidate in candidates:
            if candidate["bbox"] is not None:
                x1, y1, x2, y2 = map(int, candidate["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        frame = cv2.resize(frame, (320, 320))
        panel = np.full((370, 320, 3), 20, np.uint8)
        panel[:320] = frame
        label = "/".join(sorted({c["source_label"] for c in candidates}))
        cv2.putText(panel, label[:45], (5, 340), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1)
        cv2.putText(panel, "PENDING - NOT desk/stove ground truth", (5, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 165, 255), 1)
        panels.append(panel)
    manifest = {"status": "pending_manual_review", "images": len(entries),
                "source_checkpoint_sha256": evidence["sha256"], "entries": entries,
                "policy": "No data.yaml or ground-truth labels are exported from model predictions"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if panels:
        # Prefer one kitchen, one workstation, and context-only scene in preview.
        chosen = []
        for kind in ("kitchen_oven", "workstation_table", "workstation_context_only"):
            for i, entry in enumerate(entries):
                if any(c["candidate"] == kind for c in entry["proposals"]):
                    if i not in chosen:
                        chosen.append(i)
                    if sum(any(c["candidate"] == kind for c in entries[j]["proposals"]) for j in chosen) >= 3:
                        break
        group = [panels[i] for i in chosen[:12]]
        group += [np.zeros_like(group[0])] * ((-len(group)) % 3)
        preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview), np.vstack([np.hstack(group[i:i + 3]) for i in range(0, len(group), 3)]))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BOOTSTRAP)
    parser.add_argument("--preview", type=Path, default=PREVIEW)
    parser.add_argument("--local-images", type=Path)
    args = parser.parse_args()
    manifest = bootstrap_candidates(args.output, args.preview, args.local_images)
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"}, indent=2))


if __name__ == "__main__":
    main()
