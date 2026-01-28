"""Drawing overlays on video frames."""
from typing import List, Dict, Optional
import cv2
import numpy as np
from ..utils.mappings import cls_to_yards


def draw_overlay(
    frame: np.ndarray,
    hash_dets: List[Dict],
    number_dets: List[Dict],
    yard_value: Optional[int],
    player_dets: Optional[List[Dict]] = None,
) -> np.ndarray:
    """Draw detection overlays on a video frame.
    
    Args:
        frame: Input video frame
        hash_dets: Hash mark detections
        number_dets: Yard number detections
        yard_value: Current yardline estimate
        player_dets: Optional player detections with poses
        
    Returns:
        Frame with overlays drawn
    """
    out = frame.copy()
    
    # Draw numbers
    for d in number_dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = f"{cls_to_yards(d['cls']) or '?'} ({d['conf']:.2f})"
        cv2.putText(out, txt, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw hash lines/keypoints
    for d in hash_dets:
        if d.get("line") is not None:
            x1, y1, x2, y2 = d["line"]
            cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if d.get("keypoints") is not None:
            kp = np.array(d["keypoints"])[:2]
            for p in kp:
                cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
    
    # Draw player detections and poses
    if player_dets:
        # COCO keypoint connections for skeleton (standard 17-point pose)
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        for player in player_dets:
            x1, y1, x2, y2 = player["bbox"]
            # Draw bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange bbox
            cv2.putText(out, f"Player {player['conf']:.2f}", (x1, y1 - 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            
            # Draw keypoints and skeleton
            if player["keypoints"] is not None and len(player["keypoints"]) > 0:
                kps = player["keypoints"]  # Shape: (N, 3) where 3 = [x, y, confidence]
                
                # Draw skeleton connections
                for (idx1, idx2) in skeleton:
                    if idx1 < len(kps) and idx2 < len(kps):
                        if kps[idx1][2] > 0.3 and kps[idx2][2] > 0.3:  # Confidence threshold
                            pt1 = (int(kps[idx1][0]), int(kps[idx1][1]))
                            pt2 = (int(kps[idx2][0]), int(kps[idx2][1]))
                            cv2.line(out, pt1, pt2, (0, 255, 255), 2)  # Cyan skeleton
                
                # Draw keypoints
                for kp in kps:
                    x, y, conf = kp
                    if conf > 0.3:
                        cv2.circle(out, (int(x), int(y)), 4, (255, 0, 255), -1)  # Magenta keypoints
    
    # Yard value summary
    cv2.putText(out, f"Yardline: {yard_value if yard_value is not None else 'unknown'}", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out
