"""Player detection and pose estimation."""
from typing import List, Dict
import numpy as np
from ultralytics import YOLO


def detect_players_and_poses(
    frame: np.ndarray,
    player_model: YOLO,
    pose_model: YOLO,
    device_player: str,
    device_pose: str,
    player_conf_thresh: float = 0.5,
) -> List[Dict]:
    """Two-step player detection and pose estimation.
    
    Step 1: Detect players with bounding boxes
    Step 2: Run pose estimation on each player crop
    
    Args:
        frame: Input video frame
        player_model: YOLO model for player detection
        pose_model: YOLO model for pose estimation
        device_player: Device for player model inference
        device_pose: Device for pose model inference
        player_conf_thresh: Confidence threshold for player detection
    
    Returns:
        List of {"bbox": (x1,y1,x2,y2), "conf": float, "cls": int,
                "keypoints": np.ndarray (N,3), "crop_offset": (x1,y1)}
    """
    player_results = []
    
    # Step 1: Detect players
    player_preds = player_model.predict(source=frame, verbose=False, device=device_player, conf=player_conf_thresh)[0]
    
    if player_preds.boxes is None or len(player_preds.boxes) == 0:
        return player_results
    
    h, w = frame.shape[:2]
    
    for box in player_preds.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(box.conf[0].cpu().item())
        cls = int(box.cls[0].cpu().item())
        
        # Ensure bbox is within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Step 2: Crop player region and run pose estimation
        player_crop = frame[y1:y2, x1:x2]
        
        # Run pose model on the crop
        pose_preds = pose_model.predict(source=player_crop, verbose=False, device=device_pose)[0]
        
        keypoints = None
        if hasattr(pose_preds, 'keypoints') and pose_preds.keypoints is not None and len(pose_preds.keypoints) > 0:
            # Get the first (most confident) pose detection
            # Keypoints are in crop coordinates, need to offset back to frame coordinates
            kp_crop = pose_preds.keypoints[0].xy.cpu().numpy()  # Shape: (batch?, num_keypoints, 2)
            
            # Squeeze any extra batch dimensions
            while kp_crop.ndim > 2:
                kp_crop = kp_crop.squeeze(0)
            
            # Skip if no keypoints detected
            if kp_crop.shape[0] == 0:
                keypoints = None
            else:
                # Get confidence scores
                if hasattr(pose_preds.keypoints[0], 'conf') and pose_preds.keypoints[0].conf is not None:
                    kp_conf = pose_preds.keypoints[0].conf.cpu().numpy()
                else:
                    kp_conf = np.ones(kp_crop.shape[0])
                
                # Ensure kp_conf is 1D
                while kp_conf.ndim > 1:
                    kp_conf = kp_conf.squeeze(0)
                
                # Convert crop coordinates to frame coordinates
                kp_frame = kp_crop.copy()
                kp_frame[:, 0] += x1
                kp_frame[:, 1] += y1
                
                # Combine xy + confidence (N, 3)
                keypoints = np.concatenate([kp_frame, kp_conf.reshape(-1, 1)], axis=1)
        
        player_results.append({
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
            "cls": cls,
            "keypoints": keypoints,
            "crop_offset": (x1, y1)
        })
    
    return player_results
