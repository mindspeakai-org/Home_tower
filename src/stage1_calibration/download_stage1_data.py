"""Download the stage-one Open Images validation data and YOLO11 weights."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["Couch", "Bed", "Chair", "Desk", "Gas stove"]
EXPORT_DIR = PROJECT_ROOT / "datasets" / "stage1_zones_yolo"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODEL_NAMES = ("yolo11n.pt", "yolo11n-pose.pt")


def download_dataset() -> None:
    """Export up to 300 matching validation images with only the target labels.

    FiftyOne caches the source dataset for reuse. Existing exports are preserved;
    move an existing export before running this function again.
    """
    if EXPORT_DIR.exists() and any(EXPORT_DIR.iterdir()):
        raise FileExistsError(f"Export already exists; move it first: {EXPORT_DIR}")

    import fiftyone as fo
    import fiftyone.zoo as foz

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=CLASSES,
        only_matching=True,
        max_samples=300,
        shuffle=True,
        seed=42,
    )
    # only_matching=True restricts boxes as well as selecting matching images.
    dataset.export(
        export_dir=str(EXPORT_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=CLASSES,
        split="val",
    )
    print(f"Exported {len(dataset)} validation images to {EXPORT_DIR}")


def download_weights() -> None:
    """Download missing official checkpoints into the project weights directory."""
    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        path = WEIGHTS_DIR / name
        YOLO(str(path))
        if not path.is_file():
            raise FileNotFoundError(f"Model download did not create {path}")
        print(f"Ready: {path}")


def main() -> None:
    download_dataset()
    download_weights()


if __name__ == "__main__":
    main()
