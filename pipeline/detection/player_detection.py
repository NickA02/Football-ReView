"""Player detection and pose estimation."""
from typing import List, Dict, Tuple, Optional
import numpy as np
from ultralytics import YOLO


def _detect_players(
    frame: np.ndarray,
    player_model: YOLO,
    device: str,
    conf_thresh: float = 0.5,
) -> List[Dict]:
    """Detect players in a frame.
    
    Args:
        frame: Input video frame
        player_model: YOLO model for player detection
        device: Device for model inference
        conf_thresh: Confidence threshold for player detection
        
    Returns:
        List of {"bbox": (x1,y1,x2,y2), "conf": float, "cls": int}
    """
    player_preds = player_model.predict(source=frame, verbose=False, device=device, conf=conf_thresh)[0]
    
    if player_preds.boxes is None or len(player_preds.boxes) == 0:
        return []
    
    h, w = frame.shape[:2]
    
    # Vectorized: extract all box data at once
    boxes = player_preds.boxes.xyxy.cpu().numpy().astype(int)  # (N, 4)
    confs = player_preds.boxes.conf.cpu().numpy()
    classes = player_preds.boxes.cls.cpu().numpy().astype(int)
    
    # Vectorized: clip all boxes to frame bounds at once
    boxes[:, 0] = np.maximum(0, boxes[:, 0])  # x1
    boxes[:, 1] = np.maximum(0, boxes[:, 1])  # y1
    boxes[:, 2] = np.minimum(w, boxes[:, 2])  # x2
    boxes[:, 3] = np.minimum(h, boxes[:, 3])  # y2
    
    # Filter out invalid boxes (where x2 <= x1 or y2 <= y1)
    valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    
    # Build detection list for valid boxes
    return [{"bbox": (int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])),
             "conf": float(confs[i]),
             "cls": int(classes[i])}
            for i in range(len(boxes)) if valid_mask[i]]


def _estimate_pose_for_player(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pose_model: YOLO,
    device: str,
    buffer: int = 5,
) -> Optional[np.ndarray]:
    """Estimate pose for a detected player.
    
    Args:
        frame: Input video frame
        bbox: Player bounding box (x1, y1, x2, y2)
        pose_model: YOLO model for pose estimation
        device: Device for model inference
        buffer: Pixel buffer to add around bbox for cropping
        
    Returns:
        Keypoints array (N, 3) with [x, y, conf] in frame coordinates, or None
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    
    # Add buffer and clip to frame bounds
    x1_crop = max(0, x1 - buffer)
    y1_crop = max(0, y1 - buffer)
    x2_crop = min(w, x2 + buffer)
    y2_crop = min(h, y2 + buffer)
    
    # Crop player region with buffer
    player_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
    
    # Run pose model on the crop
    pose_preds = pose_model.predict(source=player_crop, verbose=False, device=device)[0]
    
    if not (hasattr(pose_preds, 'keypoints') and pose_preds.keypoints is not None and len(pose_preds.keypoints) > 0):
        return None
    
    # Get the first (most confident) pose detection
    kp_crop = pose_preds.keypoints[0].xy.cpu().numpy()
    
    # Squeeze any extra batch dimensions
    while kp_crop.ndim > 2:
        kp_crop = kp_crop.squeeze(0)
    
    if kp_crop.shape[0] == 0:
        return None
    
    # Get confidence scores
    if hasattr(pose_preds.keypoints[0], 'conf') and pose_preds.keypoints[0].conf is not None:
        kp_conf = pose_preds.keypoints[0].conf.cpu().numpy()
    else:
        kp_conf = np.ones(kp_crop.shape[0])
    
    # Ensure kp_conf is 1D
    while kp_conf.ndim > 1:
        kp_conf = kp_conf.squeeze(0)
    
    # Convert crop coordinates to frame coordinates (vectorized)
    kp_frame = kp_crop.copy()
    kp_frame[:, 0] += x1_crop  # Offset by crop x
    kp_frame[:, 1] += y1_crop  # Offset by crop y
    
    # Combine xy + confidence (N, 3)
    return np.concatenate([kp_frame, kp_conf.reshape(-1, 1)], axis=1)


def detect_players_and_poses(
    frame: np.ndarray,
    player_model: YOLO,
    pose_model: YOLO,
    player_device: str,
    pose_device: str,
    player_conf_thresh: float = 0.5,
    pose_conf_thresh: float = 0.7,
    crop_buffer: int = 5,
) -> List[Dict]:
    """Two-step player detection and pose estimation.
    
    Step 1: Detect players with bounding boxes
    Step 2: Run pose estimation on high-confidence player crops
    
    Args:
        frame: Input video frame
        player_model: YOLO model for player detection
        pose_model: YOLO model for pose estimation
        device: Device for model inference
        player_conf_thresh: Confidence threshold for player detection
        pose_conf_thresh: Minimum confidence to attempt pose estimation
        crop_buffer: Pixel buffer around bbox for pose cropping (default: 5)
    
    Returns:
        List of {"bbox": (x1,y1,x2,y2), "conf": float, "cls": int,
                "keypoints": np.ndarray (N,3) or None, "crop_offset": (x1,y1)}
    """
    # Step 1: Detect all players
    player_dets = _detect_players(frame, player_model, player_device, player_conf_thresh)
    
    # Step 2: Estimate poses only for high-confidence detections
    player_results = []
    for det in player_dets:
        keypoints = None
        if det["conf"] >= pose_conf_thresh:
            keypoints = _estimate_pose_for_player(
                frame, det["bbox"], pose_model, pose_device, buffer=crop_buffer
            )
        
        player_results.append({
            "bbox": det["bbox"],
            "conf": det["conf"],
            "cls": det["cls"],
            "keypoints": keypoints,
            "crop_offset": (det["bbox"][0], det["bbox"][1])
        })
    
    return player_results
