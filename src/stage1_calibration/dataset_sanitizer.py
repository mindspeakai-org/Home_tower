"""Build a detection-only shadow view without changing source annotations."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SHADOW = ROOT / "datasets/furniture_sanitized_shadow"
CACHE = ROOT / "datasets/furniture_original_cache"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_name(name: str) -> str:
    name = "_".join(name.strip().casefold().split())
    return "couch" if name in {"sofa", "couch"} else name


def detection_rows(text: str, class_count: int) -> list[list]:
    """Convert each row independently; never reinterpret bbox values as vertices."""
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        values = np.asarray([float(value) for value in line.split()])
        if (len(values) < 5 or not np.isfinite(values).all() or values[0] != int(values[0])
                or not 0 <= values[0] < class_count):
            raise ValueError(f"Invalid class/coordinates at line {number}")
        if len(values) == 5:
            _, x, y, width, height = values
        elif len(values) >= 7 and len(values) % 2 == 1:
            points = values[1:].reshape(-1, 2)
            lower, upper = points.min(axis=0), points.max(axis=0)
            x, y = (lower + upper) / 2
            width, height = upper - lower
        else:
            raise ValueError(f"Invalid polygon at line {number}")
        corners = [x - width / 2, y - height / 2, x + width / 2, y + height / 2]
        if width <= 0 or height <= 0 or min(corners) < -1e-6 or max(corners) > 1 + 1e-6:
            raise ValueError(f"Degenerate/out-of-image detection at line {number}")
        rows.append([int(values[0]), float(x), float(y), float(width), float(height)])
    return rows


def label_text(rows: list[list]) -> str:
    return "".join(f"{int(row[0])} " + " ".join(f"{v:.10g}" for v in row[1:]) + "\n" for row in rows)


def extract_original_cache(archive: Path, destination: Path = CACHE) -> dict:
    """ZIP members need one materialized image cache before they can be symlinked."""
    archive_hash = digest(archive)
    if destination.exists():
        manifest = json.loads((destination / "source_manifest.json").read_text())
        if manifest["archive_sha256"] != archive_hash:
            raise ValueError("Existing source cache belongs to a different archive")
        for name, expected in manifest["files"].items():
            if digest(destination / name) != expected:
                raise ValueError(f"Source cache changed: {name}")
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="source-cache-", dir=destination.parent))
    try:
        files = {}
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                if entry.is_dir():
                    continue
                target = temporary / entry.filename
                if not target.resolve().is_relative_to(temporary.resolve()):
                    raise ValueError("Unsafe ZIP member path")
                target.parent.mkdir(parents=True, exist_ok=True)
                data = source.read(entry)
                target.write_bytes(data)
                target.chmod(0o444)
                files[entry.filename] = hashlib.sha256(data).hexdigest()
        manifest = {"archive": str(archive.resolve()), "archive_sha256": archive_hash, "files": files}
        (temporary / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
        temporary.rename(destination)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def sanitize_dataset(archive: Path, output: Path = SHADOW, cache: Path = CACHE) -> dict:
    source = extract_original_cache(archive, cache)
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["archive_sha256"] != source["archive_sha256"]:
            raise ValueError("Shadow directory belongs to a different source")
        return manifest
    names_line = next(line for line in (cache / "data.yaml").read_text().splitlines()
                      if line.startswith("names:"))
    original_names = ast.literal_eval(names_line.split(":", 1)[1].strip())
    names = [canonical_name(name) for name in original_names]
    if len(set(names)) != len(names):
        raise ValueError("Aliases collide; an explicit reviewed ID mapping is required")
    output.mkdir(parents=True)
    counts = {}
    for split in ("train", "valid", "test"):
        image_dir, labels_dir = output / split / "images", output / split / "labels"
        image_dir.mkdir(parents=True)
        labels_dir.mkdir()
        counter = Counter()
        for image in sorted((cache / split / "images").iterdir()):
            label = cache / split / "labels" / (image.stem + ".txt")
            rows = detection_rows(label.read_text(), len(names))
            (labels_dir / label.name).write_text(label_text(rows))
            (image_dir / image.name).symlink_to(image.resolve())
            counter.update(images=1, annotations=len(rows), empty=int(not rows))
        counts[split] = dict(counter)
    manifest = {"archive_sha256": source["archive_sha256"], "original_names": original_names,
                "names": names, "class_ids_preserved": True, "review_status": "unreviewed",
                "source_cache": str(cache.resolve()), "splits": counts}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # JSON is a valid YAML subset, supported by the native dataset loader.
    config = {"path": str(output.resolve()), "train": "train/images", "val": "valid/images",
              "test": "test/images", "nc": len(names), "names": names}
    (output / "data.yaml").write_text(json.dumps(config, indent=2))
    return manifest


def verify_native_dataset(output: Path = SHADOW) -> dict:
    from ultralytics.data.dataset import YOLODataset

    manifest = json.loads((output / "manifest.json").read_text())
    results = {}
    for split, expected in manifest["splits"].items():
        dataset = YOLODataset(img_path=str(output / split / "images"), imgsz=480,
                              data={"names": dict(enumerate(manifest["names"]))},
                              task="detect", augment=False, cache=False)
        cached = np.load(output / split / "labels.cache", allow_pickle=True).item()
        found, missing, empty, corrupt, total = map(int, cached["results"])
        if len(dataset) != expected["images"] or missing or corrupt:
            raise ValueError(f"Native loader rejected samples in {split}: {cached['results']}")
        results[split] = dict(loaded=len(dataset), found=found, missing=missing, empty=empty,
                              corrupt=corrupt, total=total,
                              annotations=sum(len(label["cls"]) for label in dataset.labels))
    (output / "native_validation.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=next((ROOT / "data").glob("*.zip")))
    parser.add_argument("--output", type=Path, default=SHADOW)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    sanitize_dataset(args.archive, args.output, args.cache)
    print(json.dumps(verify_native_dataset(args.output), indent=2))


if __name__ == "__main__":
    main()
