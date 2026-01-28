"""Field element detection (hash marks and yard numbers)."""
from typing import List, Dict, Tuple
import numpy as np
from ultralytics import YOLO


def _detect_hash_marks(frame: np.ndarray, hash_model: YOLO, device: str) -> List[Dict]:
    """Detect hash marks on a frame.
    
    Args:
        frame: Input video frame
        hash_model: YOLO model for hash mark detection
        device: Device for model inference
        
    Returns:
        List of {"keypoints": Optional[np.ndarray], 
                "line": Optional[Tuple[int,int,int,int]], "conf": float}
    """
    hres = hash_model.predict(source=frame, verbose=False, device=device)[0]
    
    # Case 1: Model outputs keypoints (hash marks are keypoint detections)
    if hasattr(hres, "keypoints") and hres.keypoints is not None and len(hres.keypoints) > 0:
        confs = hres.boxes.conf.tolist() if hres.boxes is not None else [0.0] * len(hres.keypoints.xy)
        # Vectorized: build all dicts at once using zip and list comprehension
        # Filter out detections where any keypoint is at (0, 0) - indicates low confidence
        results = []
        for kp, conf in zip(hres.keypoints.xy, confs):
            kp_array = np.array(kp)
            # Check if any keypoint is at origin (0, 0) - invalid detection
            if len(kp_array) > 0 and not np.any(np.all(kp_array == 0, axis=1)):
                results.append({"keypoints": kp, "line": None, "conf": float(conf)})
        return results
    
    # Case 2: Model outputs boxes - convert to centerline approximation (fallback)
    elif hres.boxes is not None:
        # Vectorized: extract all box data at once
        boxes = hres.boxes.xyxy.cpu().numpy()  # (N, 4) array
        confs = hres.boxes.conf.cpu().numpy() if hasattr(hres.boxes, "conf") else np.zeros(len(boxes))
        
        # Compute centerlines for all boxes at once
        cx = ((boxes[:, 0] + boxes[:, 2]) / 2).astype(int)
        y1 = boxes[:, 1].astype(int)
        y2 = boxes[:, 3].astype(int)
        
        # Build list of detections
        return [{"keypoints": None, "line": (int(cx[i]), int(y1[i]), int(cx[i]), int(y2[i])), "conf": float(confs[i])}
                for i in range(len(boxes))]
    
    return []


def _detect_yard_numbers(frame: np.ndarray, number_model: YOLO, device: str) -> List[Dict]:
    """Detect yard numbers on a frame.
    
    Args:
        frame: Input video frame
        number_model: YOLO model for yard number detection
        device: Device for model inference
        
    Returns:
        List of {"bbox": (x1, y1, x2, y2), "cls": int, "conf": float}
    """
    nres = number_model.predict(source=frame, verbose=False, device=device)[0]
    
    if nres.boxes is None or len(nres.boxes) == 0:
        return []
    
    # Vectorized: extract all data at once
    boxes = nres.boxes.xyxy.cpu().numpy().astype(int)  # (N, 4) array
    confs = nres.boxes.conf.cpu().numpy() if hasattr(nres.boxes, "conf") else np.zeros(len(boxes))
    classes = nres.boxes.cls.cpu().numpy().astype(int) if hasattr(nres.boxes, "cls") else np.full(len(boxes), -1)
    
    # Build list of detections using vectorized data
    return [{"bbox": (int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])),
             "cls": int(classes[i]),
             "conf": float(confs[i])}
            for i in range(len(boxes))]


def detect_hash_and_numbers(
    frame: np.ndarray,
    hash_model: YOLO,
    number_model: YOLO,
    hash_device: str,
    number_device: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Run hash mark and yard number detectors on a frame.

    Args:
        frame: Input video frame
        hash_model: YOLO model for hash mark detection
        number_model: YOLO model for yard number detection
        device: Device for model inference

    Returns:
        Tuple of:
        - hash_dets: list of {"keypoints": Optional[np.ndarray], 
                             "line": Optional[Tuple[int,int,int,int]], "conf": float}
        - number_dets: list of {"bbox": (x1, y1, x2, y2), "cls": int, "conf": float}
    """
    hash_dets = _detect_hash_marks(frame, hash_model, hash_device)
    number_dets = _detect_yard_numbers(frame, number_model, number_device)
    return hash_dets, number_dets
