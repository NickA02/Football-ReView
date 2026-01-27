"""
Quick test script for player detection and pose estimation.
Tests the two-step approach on a single frame.
"""

import cv2
import numpy as np
from pathlib import Path
import typer
from ultralytics import YOLO
from rich import print


def main(
    video: Path = typer.Argument(..., help="Input video file"),
    player_model: Path = typer.Argument(..., help="Player detection model (e.g., yolov8n.pt for COCO)"),
    pose_model: Path = typer.Argument(..., help="Pose estimation model (e.g., yolov8n-pose.pt)"),
    frame_num: int = typer.Option(0, help="Frame number to test (0 = first frame)"),
    output: Path = typer.Option("test_player_detection.jpg", help="Output image path"),
    player_conf: float = typer.Option(0.5, help="Player detection confidence threshold"),
    device: str = typer.Option("auto", help="Device: 'auto'|'cpu'|'mps'|'0' for CUDA"),
):
    """
    Test player detection and pose estimation on a single frame.
    
    Example:
        python src/test_player_detection.py 10_SL.mp4 \\
            models/player_detection_yolov8n.pt \\
            models/player_pose_yolov8n-pose.pt \\
            --frame-num 100 --output test_frame.jpg
    """
    print(f"[bold]Loading models...[/bold]")
    player_det_model = YOLO(str(player_model))
    pose_est_model = YOLO(str(pose_model))
    
    # Auto-detect device
    import torch
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"
    
    # Check for MPS pose bug
    pose_device = device
    if device == "mps" and pose_est_model.task == "pose":
        print("[yellow]MPS detected with pose model - routing pose to CPU[/yellow]")
        pose_device = "cpu"
    
    print(f"Using devices -> player: {device}, pose: {pose_device}")
    
    # Load video
    print(f"\n[bold]Loading video: {video}[/bold]")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[red]Error: Could not open video {video}[/red]")
        return
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"[red]Error: Could not read frame {frame_num}[/red]")
        return
    
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    
    # Step 1: Detect players
    print(f"\n[bold cyan]Step 1: Detecting players...[/bold cyan]")
    player_results = player_det_model.predict(
        source=frame, verbose=False, device=device, conf=player_conf
    )[0]
    
    if player_results.boxes is None or len(player_results.boxes) == 0:
        print("[yellow]No players detected![/yellow]")
        return
    
    print(f"✅ Detected {len(player_results.boxes)} players")
    
    # Step 2: Run pose on each player
    print(f"\n[bold cyan]Step 2: Estimating poses for each player...[/bold cyan]")
    
    out_frame = frame.copy()
    
    # COCO skeleton connections
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    player_count = 0
    pose_count = 0
    
    for box in player_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(box.conf[0].cpu().item())
        
        # Ensure bbox is within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        player_count += 1
        
        # Draw bounding box
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(
            out_frame, f"Player {conf:.2f}", (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2
        )
        
        # Crop and run pose
        player_crop = frame[y1:y2, x1:x2]
        pose_results = pose_est_model.predict(
            source=player_crop, verbose=False, device=pose_device
        )[0]
        
        if hasattr(pose_results, 'keypoints') and pose_results.keypoints is not None and len(pose_results.keypoints) > 0:
            pose_count += 1
            
            try:
                # Get keypoints (in crop coordinates)
                kp_crop = pose_results.keypoints[0].xy.cpu().numpy()
                print(f"  [dim]kp_crop shape: {kp_crop.shape}, dtype: {kp_crop.dtype}[/dim]")
                
                # Squeeze any extra dimensions (batch dimension)
                while kp_crop.ndim > 2:
                    kp_crop = kp_crop.squeeze(0)
                print(f"  [dim]kp_crop after squeeze: {kp_crop.shape}[/dim]")
                
                # Skip if no keypoints detected
                if kp_crop.shape[0] == 0:
                    print(f"  [yellow]No keypoints detected for this player[/yellow]")
                    continue
                
                # Get confidence scores
                if hasattr(pose_results.keypoints[0], 'conf') and pose_results.keypoints[0].conf is not None:
                    kp_conf = pose_results.keypoints[0].conf.cpu().numpy()
                else:
                    kp_conf = np.ones(kp_crop.shape[0])
                
                print(f"  [dim]kp_conf shape: {kp_conf.shape}, dtype: {kp_conf.dtype}[/dim]")
                
                # Ensure kp_conf is 1D array and matches keypoints
                while kp_conf.ndim > 1:
                    kp_conf = kp_conf.squeeze(0)
                print(f"  [dim]kp_conf after squeeze: {kp_conf.shape}[/dim]")
                
                # Transform to frame coordinates
                kp_frame = kp_crop.copy()
                kp_frame[:, 0] += x1
                kp_frame[:, 1] += y1
                
                print(f"  [dim]Drawing skeleton with {len(kp_frame)} keypoints...[/dim]")
                
                # Draw skeleton
                for (idx1, idx2) in skeleton:
                    try:
                        if idx1 < len(kp_frame) and idx2 < len(kp_frame) and idx1 < len(kp_conf) and idx2 < len(kp_conf):
                            conf1 = float(kp_conf[idx1])
                            conf2 = float(kp_conf[idx2])
                            
                            if conf1 > 0.3 and conf2 > 0.3:
                                pt1 = (int(kp_frame[idx1, 0]), int(kp_frame[idx1, 1]))
                                pt2 = (int(kp_frame[idx2, 0]), int(kp_frame[idx2, 1]))
                                
                                cv2.line(out_frame, pt1, pt2, (0, 255, 255), 2)
                    except Exception as e:
                        print(f"  [yellow]Warning: Failed to draw skeleton connection ({idx1}, {idx2}): {e}[/yellow]")
                        continue
                
                print(f"  [dim]Drawing {len(kp_frame)} keypoint circles...[/dim]")
                
                # Draw keypoints
                for i in range(len(kp_frame)):
                    try:
                        if i < len(kp_conf):
                            conf = float(kp_conf[i])
                            
                            if conf > 0.3:
                                x_coord = int(kp_frame[i, 0])
                                y_coord = int(kp_frame[i, 1])
                                
                                cv2.circle(out_frame, (x_coord, y_coord), 4, (255, 0, 255), -1)
                    except Exception as e:
                        print(f"  [yellow]Warning: Failed to draw keypoint {i}: {e}[/yellow]")
                        continue
                        
            except Exception as e:
                print(f"  [red]Error processing pose for player {player_count}: {e}[/red]")
                import traceback
                traceback.print_exc()
    
    print(f"✅ Processed {player_count} players, {pose_count} poses estimated")
    
    # Save output
    cv2.imwrite(str(output), out_frame)
    print(f"\n[bold green]✅ Saved result to: {output}[/bold green]")
    print(f"\n[dim]Legend:[/dim]")
    print(f"  [dim]Orange boxes: Player detections[/dim]")
    print(f"  [dim]Magenta dots: Pose keypoints[/dim]")
    print(f"  [dim]Cyan lines: Skeleton connections[/dim]")


if __name__ == "__main__":
    typer.run(main)
