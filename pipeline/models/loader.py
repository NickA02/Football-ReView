"""Model loading utilities."""
from typing import Optional, Tuple
from ultralytics import YOLO
from rich import print


def load_models(
    hash_model_path: str,
    number_model_path: str,
    player_model_path: Optional[str] = None,
    pose_model_path: Optional[str] = None,
) -> Tuple[YOLO, YOLO, Optional[YOLO], Optional[YOLO]]:
    """Load YOLO models for detection tasks.
    
    Args:
        hash_model_path: Path to hash mark detection model
        number_model_path: Path to yard number detection model
        player_model_path: Optional path to player detection model
        pose_model_path: Optional path to pose estimation model
        
    Returns:
        Tuple of (hash_model, number_model, player_model, pose_model)
    """
    print(f"[bold]Loading hash model from:[/bold] {hash_model_path}")
    hash_model = YOLO(hash_model_path)

    number_model = YOLO(number_model_path)
    player_model = YOLO(player_model_path) if player_model_path else None
    pose_model = YOLO(pose_model_path) if pose_model_path else None
    return hash_model, number_model, player_model, pose_model
