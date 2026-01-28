"""Main pipeline execution script."""
import json
from pathlib import Path
from typing import Optional, Dict
import cv2
import numpy as np
import typer
from rich import print

from pipeline.models import load_models
from pipeline.detection import detect_hash_and_numbers, detect_players_and_poses
from pipeline.processing import fuse_detections
from pipeline.geometry import estimate_homography, homography_quality, smooth_homography
from pipeline.visualization import draw_overlay
from pipeline.utils import resolve_device


def _project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project points using homography matrix."""
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts.astype(np.float32), ones], axis=1)
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


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
    """Run the football field analysis pipeline on a video."""
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
    anchor_cache = []  # Each entry: {"line": {...}, "frame": int, "age": int}
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
        from pipeline.utils.mappings import cls_to_yards
        
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
