"""Deterministic test review packs and clean copies gated by operator decisions."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .dataset_sanitizer import ROOT, SHADOW, detection_rows, digest, label_text


REVIEW = ROOT / "reports/stage1_validation/review_subset"


def create_review_subset(output: Path = REVIEW, count: int = 125, seed: int = 20261114) -> dict:
    if not 100 <= count <= 150:
        raise ValueError("The stratified review pack must contain 100–150 images")
    if output.exists():
        raise FileExistsError("Review packs are immutable; choose a new path to preserve operator decisions")
    inventory = json.loads((ROOT / "reports/furniture_audit/inventory.json").read_text())
    source = json.loads((SHADOW / "manifest.json").read_text())
    predictions = [json.loads(line) for line in
                   (ROOT / "reports/furniture_audit/test_predictions.jsonl").read_text().splitlines()]
    names = source["names"]

    def order(record):
        return hashlib.sha256(f"{seed}:{record['image']}".encode()).hexdigest()

    records = sorted(predictions, key=order)
    by_name = {record["image"]: record for record in records}
    selected, reasons = [], {}

    def include(record, reason):
        if record["image"] not in reasons:
            selected.append(record)
            reasons[record["image"]] = []
        reasons[record["image"]].append(reason)

    for identifier in range(len(names)):
        candidates = [record for record in records if any(int(g[0]) == identifier for g in record["ground_truth"])]
        for record in candidates[:3]:
            include(record, f"class_{identifier}")
    for name in inventory["mixed_format_files"]:
        if name.startswith("test/"):
            include(by_name[name.replace("/labels/", "/images/").replace(".txt", ".jpg")], "mixed_format")
    empty = [r for r in records if not r["ground_truth"]]
    for record in empty[:10]:
        include(record, "empty_original_annotations")
    examples = json.loads((ROOT / "reports/furniture_audit/indoor_examples.json").read_text())
    for record in examples:
        if record["index"] in {"02", "03", "12", "16", "17", "18", "19", "21", "22", "23"}:
            include(by_name[record["image"]], "known_annotation_or_model_problem")
    if len(selected) > count:
        raise ValueError(f"Required strata need {len(selected)} images; increase count within 150")
    for record in records:
        if len(selected) >= count:
            break
        if record["image"] not in reasons:
            include(record, "deterministic_fill")
    output.mkdir(parents=True)
    (output / "decisions").mkdir()
    (output / "overlays").mkdir()
    items, panels = [], []
    coverage = Counter()
    for index, record in enumerate(selected):
        image = SHADOW / record["image"]
        label = SHADOW / "test/labels" / (image.stem + ".txt")
        rows = detection_rows(label.read_text(), len(names))
        key = f"{index:03d}"
        decision = {"id": key, "image_sha256": digest(image), "status": "pending",
                    "reviewer": "", "reviewed_at": "", "boxes": rows, "notes": ""}
        (output / "decisions" / (key + ".json")).write_text(json.dumps(decision, indent=2))
        entry = {"id": key, "image": str(image.resolve()), "archive_member": record["image"],
                 "image_sha256": digest(image), "original_label_sha256": digest(
                     Path(source["source_cache"]) / "test/labels" / label.name),
                 "sanitized_original_boxes": rows, "reasons": reasons[record["image"]]}
        items.append(entry)
        coverage.update(int(row[0]) for row in rows)
        frame = cv2.imread(str(image))
        overlay = draw_boxes(frame, rows, names)
        overlay = cv2.copyMakeBorder(overlay, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(overlay, f"{key} PENDING - original annotations, not verified GT", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv2.imwrite(str(output / "overlays" / (key + ".jpg")), overlay)
        panels.append(cv2.resize(overlay, (312, 335)))
    for start in range(0, len(panels), 20):
        group = panels[start:start + 20]
        group += [np.zeros_like(group[0])] * ((-len(group)) % 4)
        sheet = np.vstack([np.hstack(group[i:i + 4]) for i in range(0, len(group), 4)])
        cv2.imwrite(str(output / f"review_sheet_{start // 20 + 1}.jpg"), sheet)
    manifest = {"seed": seed, "count": len(items), "archive_sha256": source["archive_sha256"],
                "class_names": names, "original_class_names": source["original_names"],
                "test_annotation_coverage": dict(coverage),
                "classes_absent_in_test": [i for i in range(len(names)) if not coverage[i]],
                "policy": "Source labels only. Pending predictions are never ground truth.", "items": items}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def draw_boxes(frame, rows, names):
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    for identifier, x, y, w, h in rows:
        x1, y1, x2, y2 = map(int, ((x - w / 2) * width, (y - h / 2) * height,
                                  (x + w / 2) * width, (y + h / 2) * height))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 1)
        cv2.putText(canvas, f"{int(identifier)} {names[int(identifier)]}", (x1, max(13, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 220, 0), 1)
    return canvas


def reviewed_items(review_dir: Path) -> tuple[dict, list]:
    manifest = json.loads((review_dir / "manifest.json").read_text())
    accepted = []
    for item in manifest["items"]:
        decision = json.loads((review_dir / "decisions" / (item["id"] + ".json")).read_text())
        if decision.get("id") != item["id"]:
            raise ValueError("Review decision ID does not match its manifest")
        status = decision.get("status")
        if status not in {"pending", "rejected", "accepted", "corrected"}:
            raise ValueError(f"Invalid review status: {item['id']}")
        if status in {"pending", "rejected"}:
            continue
        if not str(decision.get("reviewer", "")).strip() or not decision.get("reviewed_at"):
            raise ValueError(f"Missing manual review metadata for {item['id']}")
        if datetime.fromisoformat(decision["reviewed_at"]).tzinfo is None:
            raise ValueError("Review time must include a timezone")
        if decision.get("image_sha256") != item["image_sha256"] or digest(Path(item["image"])) != item["image_sha256"]:
            raise ValueError("Image changed since review pack creation")
        rows = decision["boxes"]
        if any(not isinstance(row, list) or len(row) != 5 for row in rows):
            raise ValueError("Reviewed annotations must be detection boxes")
        parsed = detection_rows("\n".join(" ".join(map(str, row)) for row in rows), len(manifest["class_names"]))
        if status == "accepted" and (len(parsed) != len(item["sanitized_original_boxes"]) or not np.allclose(
                np.asarray(parsed), np.asarray(item["sanitized_original_boxes"]))):
            raise ValueError("Changed labels need status corrected, not accepted")
        accepted.append((item, {**decision, "boxes": parsed}))
    return manifest, accepted


def build_clean_evaluation(review_dir: Path, output: Path) -> dict:
    manifest, accepted = reviewed_items(review_dir)
    if not accepted:
        return {"status": "unavailable", "reviewed_images": 0,
                "reason": "No manually accepted/corrected annotations; predictions cannot supply ground truth"}
    if output.exists():
        raise FileExistsError("Clean evaluation copies are immutable; choose a new output version")
    (output / "test/images").mkdir(parents=True)
    (output / "test/labels").mkdir()
    approved = []
    for item, decision in accepted:
        image = Path(item["image"])
        (output / "test/images" / image.name).symlink_to(image.resolve())
        (output / "test/labels" / (image.stem + ".txt")).write_text(label_text(decision["boxes"]))
        approved.append({**item, "decision": decision})
    config = {"path": str(output.resolve()), "test": "test/images", "val": "test/images",
              "nc": len(manifest["class_names"]), "names": manifest["class_names"]}
    (output / "data.yaml").write_text(json.dumps(config, indent=2))
    record = {"status": "reviewed_copy", "reviewed_images": len(approved),
              "archive_sha256": manifest["archive_sha256"], "items": approved}
    (output / "review_manifest.json").write_text(json.dumps(record, indent=2))
    return record


def review_interactively(review_dir: Path, reviewer: str):
    """A/R accept/reject; drag adds a box; C clears; X deletes; [/] change class."""
    if not reviewer.strip():
        raise ValueError("Reviewer is required")
    manifest = json.loads((review_dir / "manifest.json").read_text())
    index, selected_class, start = 0, 0, None
    window = "Annotation review"
    current = None
    frame = None

    def mouse(event, x, y, flags, _):
        nonlocal start
        if event == cv2.EVENT_LBUTTONDOWN:
            start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and start is not None:
            height, width = frame.shape[:2]
            x1, x2 = sorted(np.clip([start[0], x], 0, width).tolist())
            y1, y2 = sorted(np.clip([start[1], y], 0, height).tolist())
            if x2 > x1 and y2 > y1:
                current["boxes"].append([selected_class, (x1 + x2) / 2 / width,
                                         (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height])
            start = None

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse)
    try:
        while 0 <= index < len(manifest["items"]):
            item = manifest["items"][index]
            path = review_dir / "decisions" / (item["id"] + ".json")
            current = json.loads(path.read_text())
            frame = cv2.imread(item["image"])
            while True:
                canvas = draw_boxes(frame, current["boxes"], manifest["class_names"])
                title = f"{item['id']} [{selected_class}] {manifest['class_names'][selected_class]} A accept R reject C clear X pop N/P skip Q quit"
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (20, 20, 20), -1)
                cv2.putText(canvas, title, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 255, 255), 1)
                cv2.imshow(window, canvas)
                key = cv2.waitKey(20) & 0xFF
                if key == ord("q"):
                    return
                if key in (ord("a"), ord("r")):
                    changed = current["boxes"] != item["sanitized_original_boxes"]
                    current.update(status=("corrected" if changed else "accepted") if key == ord("a") else "rejected",
                                   reviewer=reviewer, reviewed_at=datetime.now(timezone.utc).isoformat())
                    path.write_text(json.dumps(current, indent=2))
                    index += 1
                    break
                if key in (ord("n"), ord("p")):
                    index += 1 if key == ord("n") else -1
                    break
                if key == ord("c"):
                    current["boxes"] = []
                if key == ord("x") and current["boxes"]:
                    current["boxes"].pop()
                if key in (ord("["), ord("]")):
                    selected_class = (selected_class + (1 if key == ord("]") else -1)) % len(manifest["class_names"])
    finally:
        cv2.destroyWindow(window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--count", type=int, default=125)
    parser.add_argument("--reviewer")
    parser.add_argument("--build-clean", type=Path)
    args = parser.parse_args()
    if args.create:
        print(json.dumps({k: v for k, v in create_review_subset(args.review_dir, args.count).items() if k != "items"}, indent=2))
    elif args.build_clean:
        print(json.dumps(build_clean_evaluation(args.review_dir, args.build_clean), indent=2))
    elif args.reviewer:
        review_interactively(args.review_dir, args.reviewer)
    else:
        parser.error("Choose --create, --reviewer, or --build-clean")


if __name__ == "__main__":
    main()
