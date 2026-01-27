"""YOLOv8 trainer for hash mark keypoints.

Note: Ensure your dataset's data.yaml is configured for keypoints with 'kpt_shape', 'kpt_label' or the Ultralytics keypoints format.

Usage:
    python src/train_hash_keypoints.py \
        --data datasets/field-hash-keypoints/data.yaml \
        --model yolov8n-pose.pt \
        --imgsz 1280 \
        --epochs 50 \
        --project runs/hash_keypoints \
        --name exp1 \
        --export models/hash.pt
"""

from pathlib import Path
from typing import Optional
import typer
from ultralytics import YOLO
import torch


def main(
    data: Path = typer.Option(..., help="Path to data.yaml for hash keypoints"),
    model: str = typer.Option("yolov8n-pose.pt", help="Base YOLOv8 pose model (e.g., yolov8n-pose.pt)"),
    imgsz: int = typer.Option(1280, help="Training image size"),
    epochs: int = typer.Option(50, help="Training epochs"),
    batch: Optional[int] = typer.Option(16, help="Batch size"),
    device: str = typer.Option("auto", help="Device: 'auto'|'cpu'|'mps'|'0' for CUDA"),
    project: Optional[Path] = typer.Option(None, help="Training runs project directory"),
    name: Optional[str] = typer.Option(None, help="Experiment name"),
    export: Optional[Path] = typer.Option(Path("models/hash.pt"), help="Exported weights .pt path after training"),
):
    # Resolve device
    if device == "auto":
        if torch.backends.mps.is_available():
            resolved_device = "mps"
        elif torch.cuda.is_available():
            resolved_device = "0"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = device

    yolo = YOLO(model)
    # Ultralytics uses task inference from the model; pose models trigger keypoint training.
    train_kwargs = {
        "data": str(data),
        "imgsz": imgsz,
        "epochs": epochs,
    "device": resolved_device,
        "project": str(project) if project else None,
        "name": name,
        "exist_ok": True,
    }
    if batch is not None:
        train_kwargs["batch"] = batch
    yolo.train(**train_kwargs)

    # Resolve best weights path for pose training
    if project and name:
        best = Path(project) / name / "weights" / "best.pt"
    else:
        # Default Ultralytics path for pose: runs/pose/train/weights/best.pt
        best = Path("runs/pose/train/weights/best.pt")

    if export:
        export.parent.mkdir(parents=True, exist_ok=True)
        m = YOLO(str(best))
        m.save(str(export))
        print(f"Exported weights to {export}")


if __name__ == "__main__":
    typer.run(main)
