"""Field element detection (hash marks and yard numbers)."""
from typing import List, Dict, Tuple
import numpy as np
from ultralytics import YOLO


def detect_hash_and_numbers(
    frame: np.ndarray,
    hash_model: YOLO,
    number_model: YOLO,
    device_hash: str,
    device_numbers: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Run hash mark and yard number detectors on a frame.

    Args:
        frame: Input video frame
        hash_model: YOLO model for hash mark detection
        number_model: YOLO model for yard number detection
        device_hash: Device for hash model inference
        device_numbers: Device for number model inference

    Returns:
        Tuple of:
        - hash_dets: list of {"keypoints": Optional[np.ndarray], 
                             "line": Optional[Tuple[int,int,int,int]], "conf": float}
        - number_dets: list of {"bbox": (x1, y1, x2, y2), "cls": int, "conf": float}
    """
    # Hash marks: model may provide boxes, keypoints, or lines; we standardize.
    hash_dets = []
    hres = hash_model.predict(source=frame, verbose=False, device=device_hash)[0]
    if hasattr(hres, "keypoints") and hres.keypoints is not None:
        for kp, conf in zip(hres.keypoints.xy, hres.boxes.conf.tolist() if hres.boxes is not None else [0.0] * len(hres.keypoints.xy)):
            hash_dets.append({"keypoints": kp, "line": None, "conf": float(conf)})
    elif hres.boxes is not None:
        for b in hres.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
            # Approximate line as the long axis centerline
            cx = (x1 + x2) / 2
            hash_dets.append({"keypoints": None, "line": (int(cx), int(y1), int(cx), int(y2)), "conf": conf})

    number_dets = []
    nres = number_model.predict(source=frame, verbose=False, device=device_numbers)[0]
    if nres.boxes is not None:
        for b in nres.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
            cls = int(b.cls[0].item()) if hasattr(b, "cls") else -1
            number_dets.append({"bbox": (int(x1), int(y1), int(x2), int(y2)), "cls": cls, "conf": conf})
    return hash_dets, number_dets
