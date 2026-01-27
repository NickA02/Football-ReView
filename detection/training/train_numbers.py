"""YOLOv8 trainer for yard numbers (bounding box detection).

Usage:
    python src/train_numbers.py \
        --data datasets/field-yard-numbers/data.yaml \
        --model yolov8n.pt \
        --imgsz 1280 \
        --epochs 50 \
        --project runs/yard_numbers \
        --name exp1 \
        --export models/numbers.pt
"""

from pathlib import Path
from typing import Optional
import typer
from ultralytics import YOLO
import torch


def main(
    data: Path = typer.Option(..., help="Path to data.yaml for yard numbers"),
    model: str = typer.Option("yolov8n.pt", help="Base YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt)"),
    imgsz: int = typer.Option(1280, help="Training image size"),
    epochs: int = typer.Option(50, help="Training epochs"),
    batch: Optional[int] = typer.Option(16, help="Batch size"),
    device: str = typer.Option("auto", help="Device: 'auto'|'cpu'|'mps'|'0' for CUDA"),
    project: Optional[Path] = typer.Option(None, help="Training runs project directory"),
    name: Optional[str] = typer.Option(None, help="Experiment name"),
    export: Optional[Path] = typer.Option(Path("models/numbers.pt"), help="Exported weights .pt path after training"),
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

    # Find best weights from run directory
    if project and name:
        best = Path(project) / name / "weights" / "best.pt"
    else:
        # Ultralytics default: runs/detect/train/weights/best.pt (for task=detect)
        best = Path("runs/detect/train/weights/best.pt")

    if export:
        # Copy or re-export to a known path
        # For Ultralytics 'export' is for format conversion; here we just save the .pt file.
        export.parent.mkdir(parents=True, exist_ok=True)
        # Load and save to target path to ensure model graph is self-contained
        m = YOLO(str(best))
        m.save(str(export))
        print(f"Exported weights to {export}")


if __name__ == "__main__":
    typer.run(main)
