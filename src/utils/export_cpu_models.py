"""Export fixed-size, batch-one CPU models with an ONNX fallback."""

import logging
import platform
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.stage1_calibration.download_stage1_data import MODEL_NAMES, WEIGHTS_DIR


IMAGE_SIZE = 480
LOGGER = logging.getLogger(__name__)


def is_x86_64() -> bool:
    return platform.machine().lower() in {"x86_64", "amd64"}


def model_task(name: str) -> str:
    return "pose" if "pose" in name else "detect"


def validate_model(path: Path, task: str) -> YOLO:
    """Compile/warm the CPU backend before capture starts, including its shape."""
    model = YOLO(str(path), task=task)
    model.predict(
        np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        device="cpu", imgsz=IMAGE_SIZE, rect=False, verbose=False,
    )
    return model


def export_model(checkpoint: Path) -> Path:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Download the checkpoint first: {checkpoint}")
    if is_x86_64():
        try:
            path = Path(YOLO(str(checkpoint)).export(
                format="openvino", imgsz=IMAGE_SIZE, device="cpu", batch=1,
                dynamic=False, half=False,
            ))
            validate_model(path, model_task(checkpoint.name))
            return path
        except Exception:
            LOGGER.warning("OpenVINO export/validation failed; using ONNX", exc_info=True)

    path = Path(YOLO(str(checkpoint)).export(
        format="onnx", imgsz=IMAGE_SIZE, device="cpu", batch=1,
        dynamic=False, half=False, simplify=False, opset=17,
    ))
    validate_model(path, model_task(checkpoint.name))
    return path


def load_cpu_model(name: str) -> YOLO:
    """Prefer OpenVINO on x86; try an existing ONNX export if it cannot load."""
    stem = Path(name).stem
    candidates = []
    if is_x86_64():
        candidates.append(WEIGHTS_DIR / f"{stem}_openvino_model")
    candidates.append(WEIGHTS_DIR / f"{stem}.onnx")
    errors = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            return validate_model(path, model_task(name))
        except Exception as exc:
            LOGGER.warning("CPU backend failed for %s: %s", path, exc)
            errors.append(str(exc))
    raise RuntimeError(
        f"No usable CPU export for {name}. Run python -m src.utils.export_cpu_models. "
        + "; ".join(errors)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    for name in MODEL_NAMES:
        print(f"CPU model ready: {export_model(WEIGHTS_DIR / name)}")


if __name__ == "__main__":
    main()
