import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import typer
from rich import print


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "0"
        else:
            return "cpu"
    return device


def load_models(
    hash_model_path: str,
    number_model_path: str,
    player_model_path: Optional[str] = None,
    pose_model_path: Optional[str] = None,
) -> Tuple[YOLO, YOLO, Optional[YOLO], Optional[YOLO]]:
    print(f"[bold]Loading hash model from:[/bold] {hash_model_path}")
    hash_model = YOLO(hash_model_path)

    number_model = YOLO(number_model_path)
    player_model = YOLO(player_model_path) if player_model_path else None
    pose_model = YOLO(pose_model_path) if pose_model_path else None
    return hash_model, number_model, player_model, pose_model


def detect_hash_and_numbers(
    frame: np.ndarray,
    hash_model: YOLO,
    number_model: YOLO,
    device_hash: str,
    device_numbers: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Run detectors on a frame.

    Returns:
        hash_dets: list of {"keypoints": Optional[np.ndarray], "line": Optional[Tuple[int,int,int,int]], "conf": float}
        number_dets: list of {"bbox": (x1, y1, x2, y2), "cls": int, "conf": float}
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
    
    Returns:
        List of {"bbox": (x1,y1,x2,y2), "conf": float, "keypoints": np.ndarray (N,3), "crop_offset": (x1,y1)}
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


def cls_to_yards(cls_id: int) -> Optional[int]:
    """Map yard number class ID to actual yard value. Update if your dataset uses different IDs."""
    mapping = {
        0: 10,
        1: 20,
        2: 30,
        3: 40,
        4: 50,
    }
    return mapping.get(cls_id)


def fuse_detections(hash_dets: List[Dict], number_dets: List[Dict], template_size: Tuple[int, int]) -> Dict:
    """Fuse detections prioritizing hash lines as anchors.

    Strategy:
      - Parse hash keypoints as two points per vertical yard line: (top_hash_intersection, bottom_hash_intersection).
      - Find nearest hash line to each detected yard number and assign its absolute yard value (10,20,...,50).
      - Propagate ±5 yards to neighboring hash lines by x-order to get at least two labeled lines.
      - Build anchors: for each labeled line, add two correspondences (top/bottom hash rows) in image vs template.

    Returns dict with:
      - yard_value: Optional[int] (best yard value from numbers)
      - anchors_img: List[(x,y)]
      - anchors_field: List[(X,Y)] matching template pixels
    """
    anchors_img: List[Tuple[float, float]] = []
    anchors_field: List[Tuple[float, float]] = []

    TEMPLATE_W, TEMPLATE_H = template_size

    def yard_to_template_x(yards: int) -> float:
        return (yards / 100.0) * TEMPLATE_W

    # Hash rows (template Y) - should match your template design
    HASH_TOP_Y = TEMPLATE_H * 0.35
    HASH_BOT_Y = TEMPLATE_H * 0.65

    # Parse hash detections into lines with top/bottom points and representative x
    lines = []
    for d in hash_dets:
        if d.get("keypoints") is not None:
            kp = np.array(d["keypoints"], dtype=np.float32)
            if kp.ndim == 2 and kp.shape[0] >= 2:
                # take first two points; sort by y (top first)
                k2 = kp[:2]
                k2 = k2[np.argsort(k2[:, 1])]
                top, bot = (float(k2[0, 0]), float(k2[0, 1])), (float(k2[1, 0]), float(k2[1, 1]))
                x_line = (top[0] + bot[0]) / 2.0
                lines.append({"top": top, "bot": bot, "x": x_line, "conf": d.get("conf", 0.0)})
        elif d.get("line") is not None:
            x1, y1, x2, y2 = d["line"]
            pts = [(x1, y1), (x2, y2)]
            pts = sorted(pts, key=lambda p: p[1])
            top, bot = (float(pts[0][0]), float(pts[0][1])), (float(pts[1][0]), float(pts[1][1]))
            x_line = (top[0] + bot[0]) / 2.0
            lines.append({"top": top, "bot": bot, "x": x_line, "conf": d.get("conf", 0.0)})

    # Sort lines by x (left to right)
    lines.sort(key=lambda L: L["x"])

    # Map numbers to their yard values and centers
    num_obs = []
    for nd in number_dets:
        yv = cls_to_yards(nd.get("cls", -1))
        if yv is None:
            continue
        x1, y1, x2, y2 = nd["bbox"]
        cx = (x1 + x2) / 2.0
        num_obs.append({"cx": cx, "yard": yv, "conf": nd.get("conf", 0.0)})

    # Best yard value (for UI)
    yard_value = None
    if num_obs:
        yard_value = max(num_obs, key=lambda n: n["conf"]) ["yard"]

    # Assign yard values to nearest lines based on number centers
    for n in num_obs:
        if not lines:
            break
        nearest = min(lines, key=lambda L: abs(L["x"] - n["cx"]))
        nearest["yard"] = n["yard"]

    # If we have at least one labeled line, propagate ±5 to neighbors
    labeled_indices = [i for i, L in enumerate(lines) if "yard" in L]
    if labeled_indices:
        # Determine direction: if we have multiple labels, fit sign by their order
        direction = 1  # assume increasing yards to the right
        if len(labeled_indices) >= 2:
            i0, i1 = labeled_indices[0], labeled_indices[1]
            if lines[i1]["yard"] < lines[i0]["yard"] and lines[i1]["x"] > lines[i0]["x"]:
                direction = -1
        # Propagate from each labeled index outward
        for idx in labeled_indices:
            yard_here = lines[idx]["yard"]
            # right side
            y = yard_here
            for j in range(idx + 1, len(lines)):
                y += 5 * direction
                if "yard" not in lines[j]:
                    lines[j]["yard"] = y
                else:
                    break  # stop at next existing label
            # left side
            y = yard_here
            for j in range(idx - 1, -1, -1):
                y -= 5 * direction
                if "yard" not in lines[j]:
                    lines[j]["yard"] = y
                else:
                    break

    # Build anchors from all labeled lines with quality-based selection
    labeled_lines = [L for L in lines if "yard" in L and 0 <= L["yard"] <= 100]
    
    # Score each line by confidence and use the best ones
    # Sort by confidence descending to prioritize high-quality detections
    labeled_lines_sorted = sorted(labeled_lines, key=lambda L: L["conf"], reverse=True)
    
    if len(labeled_lines_sorted) >= 2:
        # Select up to 4 best lines, ensuring good spatial spread
        selected = []
        
        # Always take the highest confidence line
        selected.append(labeled_lines_sorted[0])
        
        # Find lines far from already selected ones (diverse spatial coverage)
        for candidate in labeled_lines_sorted[1:]:
            if len(selected) >= 4:
                break
            # Check if this candidate is spatially distinct from selected ones
            min_dist = min(abs(candidate["x"] - s["x"]) for s in selected)
            if min_dist > 50:  # at least 50px apart
                selected.append(candidate)
        
        # If we still need more and have candidates, add next best by confidence
        for candidate in labeled_lines_sorted:
            if len(selected) >= 4:
                break
            if candidate not in selected:
                selected.append(candidate)
        
        # Build anchors from selected lines
        for L in selected:
            X = yard_to_template_x(L["yard"])
            anchors_img.extend([L["top"], L["bot"]])
            anchors_field.extend([(X, HASH_TOP_Y), (X, HASH_BOT_Y)])
            
    elif len(labeled_lines) == 1:
        # one line gives two points; insufficient for H but keep for debugging/overlay
        L = labeled_lines[0]
        X = yard_to_template_x(L["yard"])
        anchors_img.extend([L["top"], L["bot"]])
        anchors_field.extend([(X, HASH_TOP_Y), (X, HASH_BOT_Y)])

    return {
        "yard_value": yard_value,
        "anchors_img": anchors_img,
        "anchors_field": anchors_field,
        "labeled_lines": labeled_lines_sorted,  # Return all for temporal cache
    }


def estimate_homography(anchors_img: List[Tuple[float, float]], anchors_field: List[Tuple[float, float]]):
    """Estimate homography using all available anchors with RANSAC."""
    if len(anchors_img) >= 4 and len(anchors_field) >= 4:
        src = np.array(anchors_img, dtype=np.float32)
        dst = np.array(anchors_field, dtype=np.float32)
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        return H, mask
    return None, None
    return None, None


def _normalize_H(H: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if H is None:
        return None
    if abs(H[2, 2]) < 1e-6:
        return H
    return H / H[2, 2]


def _project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts.astype(np.float32), ones], axis=1)
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def homography_quality(H: Optional[np.ndarray], anchors_img: List[Tuple[float, float]], anchors_field: List[Tuple[float, float]]) -> Optional[float]:
    if H is None or len(anchors_img) == 0:
        return None
    pts_img = np.array(anchors_img, dtype=np.float32)
    pts_field = np.array(anchors_field, dtype=np.float32)
    proj = _project_points(H, pts_img)
    err = np.linalg.norm(proj - pts_field, axis=1)
    return float(np.mean(err)) if err.size else None


def smooth_homography(prev_H: Optional[np.ndarray], new_H: Optional[np.ndarray], alpha: float) -> Optional[np.ndarray]:
    if new_H is None and prev_H is None:
        return None
    if prev_H is None:
        return _normalize_H(new_H)
    if new_H is None:
        return _normalize_H(prev_H)
    A = _normalize_H(prev_H).reshape(-1)
    B = _normalize_H(new_H).reshape(-1)
    C = alpha * A + (1.0 - alpha) * B
    C = C.reshape(3, 3)
    return _normalize_H(C)


def draw_overlay(
    frame: np.ndarray,
    hash_dets: List[Dict],
    number_dets: List[Dict],
    yard_value: Optional[int],
    player_dets: Optional[List[Dict]] = None,
) -> np.ndarray:
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
    cv2.putText(out, f"Yardline: {yard_value if yard_value is not None else 'unknown'}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out


def main(
    video: Path = typer.Option(..., help="Input video file"),
    hash_model: Path = typer.Option(..., help="Hash mark YOLO .pt file"),
    number_model: Path = typer.Option(..., help="Yard number YOLO .pt file"),
    field_template: Path = typer.Option(..., help="Field template image (top-down)"),
    player_model: Optional[Path] = typer.Option(None, help="Player detection YOLO .pt file (optional)"),
    pose_model: Optional[Path] = typer.Option(None, help="Pose estimation YOLO .pt file (optional, requires player_model)"),
    out: Optional[Path] = typer.Option(None, help="Optional output overlay video"),
    write_json: Optional[Path] = typer.Option(None, help="Optional JSON output for per-frame placement"),
    device: str = typer.Option("auto", help="Device: 'auto'|'cpu'|'mps'|'0' for CUDA"),
    smooth_alpha: float = typer.Option(0.8, help="EMA smoothing factor for homography (0=none, 0.8=stable)"),
    max_reproj_error: float = typer.Option(12.0, help="Reject H if mean reprojection error (px) exceeds this"),
    max_corner_drift: float = typer.Option(40.0, help="Reject H if corner drift vs last H (px) exceeds this"),
    hold_last: bool = typer.Option(True, help="Reuse last good H when current is bad/missing"),
    conf_alpha: float = typer.Option(0.85, help="EMA smoothing factor for yard-number confidences"),
    min_ema_conf: float = typer.Option(0.25, help="Minimum EMA confidence to accept a yardline class"),
    player_conf: float = typer.Option(0.5, help="Confidence threshold for player detection"),
    enable_player_poses: bool = typer.Option(True, help="Enable player detection and pose estimation"),
):
    print("[bold]Loading models...[/bold]")
    h_model, n_model, p_model, pose_m = load_models(
        str(hash_model),
        str(number_model),
        str(player_model) if player_model else None,
        str(pose_model) if pose_model else None,
    )
    
    # Validate player pose models
    if enable_player_poses:
        if p_model is None or pose_m is None:
            print("[yellow]Warning: Player poses requested but models not provided. Disabling player detection.[/yellow]")
            enable_player_poses = False
        elif player_model is None or pose_model is None:
            print("[yellow]Warning: Both --player-model and --pose-model required for player poses. Disabling.[/yellow]")
            enable_player_poses = False
    
    resolved_device = resolve_device(device)
    # Choose devices per model to avoid MPS pose bug
    hash_device = resolved_device
    numbers_device = resolved_device
    player_device = resolved_device
    pose_device = resolved_device
    
    try:
        h_task = getattr(h_model, "task", "")
    except Exception:
        h_task = ""
    if resolved_device == "mps" and h_task == "pose":
        print("[yellow]MPS pose warning detected: routing hash (pose) model to CPU; numbers remain on MPS[/yellow]")
        hash_device = "cpu"
    
    # Check pose model task for player poses
    if enable_player_poses:
        try:
            pose_task = getattr(pose_m, "task", "")
        except Exception:
            pose_task = ""
        if resolved_device == "mps" and pose_task == "pose":
            print("[yellow]MPS pose warning: routing player pose model to CPU[/yellow]")
            pose_device = "cpu"
    
    print(f"Using devices -> hash: {hash_device}, numbers: {numbers_device}")
    if enable_player_poses:
        print(f"             -> player: {player_device}, pose: {pose_device}")


    tpl = cv2.imread(str(field_template))
    if tpl is None:
        raise FileNotFoundError(f"Field template not found: {field_template}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video}")

    # Prepare writer if needed
    writer = None
    if out is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))

    results_meta = []
    frame_idx = 0
    last_H_used: Optional[np.ndarray] = None
    last_yard_value: Optional[int] = None
    # EMA state for yard-number class confidences
    ema_conf: Dict[int, float] = {10: 0.0, 20: 0.0, 30: 0.0, 40: 0.0, 50: 0.0}
    
    # Temporal cache for high-quality hash lines (for anchor refinement)
    anchor_cache: List[Dict] = []  # Each entry: {"line": {...}, "frame": int, "age": int}
    MAX_CACHE_AGE = 15  # frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hash_dets, number_dets = detect_hash_and_numbers(frame, h_model, n_model, hash_device, numbers_device)
        fused = fuse_detections(hash_dets, number_dets, (tpl.shape[1], tpl.shape[0]))
        
        # Detect players and poses if enabled
        player_dets = []
        if enable_player_poses:
            player_dets = detect_players_and_poses(
                frame, p_model, pose_m, player_device, pose_device, player_conf_thresh=player_conf
            )
        
        # Update anchor cache with new high-quality lines
        if "labeled_lines" in fused and fused["labeled_lines"]:
            for line in fused["labeled_lines"]:
                # Check if this line improves upon cached anchors
                found_better = False
                for cached in anchor_cache:
                    # Same yardline? Check if new detection is better
                    if cached["line"].get("yard") == line.get("yard"):
                        if line["conf"] > cached["line"]["conf"]:
                            cached["line"] = line
                            cached["frame"] = frame_idx
                            cached["age"] = 0
                            found_better = True
                        break
                
                if not found_better:
                    # New yardline or spatial location - add to cache
                    anchor_cache.append({"line": line, "frame": frame_idx, "age": 0})
        
        # Age out old cache entries and increment age
        anchor_cache = [c for c in anchor_cache if c["age"] < MAX_CACHE_AGE]
        for c in anchor_cache:
            c["age"] += 1

        
        # Refine anchors by merging current detections with recent cache
        # Build enhanced anchor set from cache (best lines from recent history)
        cache_lines = [c["line"] for c in sorted(anchor_cache, key=lambda x: x["line"]["conf"], reverse=True)[:6]]
        
        # Rebuild anchors using cache + current for better coverage
        if len(cache_lines) >= 2:
            TEMPLATE_W, TEMPLATE_H = tpl.shape[1], tpl.shape[0]
            HASH_TOP_Y = TEMPLATE_H * 0.35
            HASH_BOT_Y = TEMPLATE_H * 0.65
            
            refined_anchors_img = []
            refined_anchors_field = []
            
            def yard_to_template_x(yards: int) -> float:
                return (yards / 100.0) * TEMPLATE_W
            
            for line in cache_lines[:4]:  # Use up to 4 best cached lines
                if "yard" in line and 0 <= line["yard"] <= 100:
                    X = yard_to_template_x(line["yard"])
                    refined_anchors_img.extend([line["top"], line["bot"]])
                    refined_anchors_field.extend([(X, HASH_TOP_Y), (X, HASH_BOT_Y)])
            
            # If we have better anchors from cache, use them
            if len(refined_anchors_img) >= len(fused["anchors_img"]) and len(refined_anchors_img) >= 4:
                fused["anchors_img"] = refined_anchors_img
                fused["anchors_field"] = refined_anchors_field
        
        raw_H, mask = estimate_homography(fused["anchors_img"], fused["anchors_field"])

        # Quality checks
        mean_re = homography_quality(raw_H, fused["anchors_img"], fused["anchors_field"]) if raw_H is not None else None

        corner_drift = None
        if raw_H is not None and last_H_used is not None:
            h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            corners = np.array([[0, 0], [w_frame, 0], [w_frame, h_frame], [0, h_frame]], dtype=np.float32)
            p_new = _project_points(raw_H, corners)
            p_old = _project_points(last_H_used, corners)
            corner_drift = float(np.mean(np.linalg.norm(p_new - p_old, axis=1)))

        # Decide H to use
        H_used = None
        if raw_H is not None:
            accept = True
            if mean_re is not None and mean_re > max_reproj_error:
                accept = False
            if corner_drift is not None and corner_drift > max_corner_drift:
                accept = False
            if accept:
                H_used = smooth_homography(last_H_used, raw_H, alpha=smooth_alpha)
            elif hold_last:
                H_used = last_H_used
        elif hold_last:
            H_used = last_H_used

        # Yardline selection with EMA-smoothed class confidences
        # 1) build current per-class confidences (max per class in this frame)
        cur_conf: Dict[int, float] = {10: 0.0, 20: 0.0, 30: 0.0, 40: 0.0, 50: 0.0}
        for d in number_dets:
            yv = cls_to_yards(d.get("cls", -1))
            if yv is None:
                continue
            cur_conf[yv] = max(cur_conf[yv], float(d.get("conf", 0.0)))
        # 2) update EMA per class
        for y in ema_conf.keys():
            ema_conf[y] = conf_alpha * ema_conf[y] + (1.0 - conf_alpha) * cur_conf[y]
        # 3) choose best EMA class if strong enough; else fallback to per-frame fused yard
        best_yard = max(ema_conf.keys(), key=lambda k: ema_conf[k])
        yard_val = fused["yard_value"]
        if ema_conf[best_yard] >= min_ema_conf:
            yard_val = best_yard
        
        # Yardline smoothing (guard big jumps)
        if last_yard_value is not None and yard_val is not None:
            if abs(yard_val - last_yard_value) > 15:
                yard_val = last_yard_value

        if H_used is not None:
            last_H_used = H_used
        if yard_val is not None:
            last_yard_value = yard_val

        # Draw overlay for visualization
        vis = draw_overlay(frame, hash_dets, number_dets, yard_val, player_dets if enable_player_poses else None)
        if writer is not None:
            writer.write(vis)

        # Serialize player detections for JSON (convert keypoints to lists)
        player_data = []
        if enable_player_poses and player_dets:
            for p in player_dets:
                player_data.append({
                    "bbox": p["bbox"],
                    "conf": p["conf"],
                    "cls": p["cls"],
                    "keypoints": p["keypoints"].tolist() if p["keypoints"] is not None else None,
                })

        results_meta.append({
            "frame": frame_idx,
            "yard_value": yard_val,
            "yard_value_ema": best_yard,
            "ema_conf": ema_conf.copy(),
            "anchors_img": fused["anchors_img"],
            "anchors_field": fused["anchors_field"],
            "homography": H_used.tolist() if H_used is not None else None,
            "raw_h": raw_H.tolist() if raw_H is not None else None,
            "mean_reproj_error": mean_re,
            "corner_drift": corner_drift,
            "players": player_data,
        })

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if write_json is not None:
        with open(write_json, "w") as f:
            json.dump(results_meta, f, indent=2)
        print(f"Saved JSON: {write_json}")

    print("[green]Done.[/green]")


if __name__ == "__main__":
    typer.run(main)
