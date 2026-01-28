"""Homography estimation and refinement."""
from typing import List, Tuple, Optional
import cv2
import numpy as np


def estimate_homography(
    anchors_img: List[Tuple[float, float]], 
    anchors_field: List[Tuple[float, float]]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate homography using all available anchors with RANSAC.
    
    Args:
        anchors_img: Image space anchor points
        anchors_field: Field template space anchor points
        
    Returns:
        Tuple of (homography_matrix, inlier_mask)
    """
    if len(anchors_img) >= 4 and len(anchors_field) >= 4:
        src = np.array(anchors_img, dtype=np.float32)
        dst = np.array(anchors_field, dtype=np.float32)
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        return H, mask
    return None, None


def _normalize_H(H: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalize homography matrix by H[2,2]."""
    if H is None:
        return None
    if abs(H[2, 2]) < 1e-6:
        return H
    return H / H[2, 2]


def _project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project points using homography matrix."""
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts.astype(np.float32), ones], axis=1)
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def homography_quality(
    H: Optional[np.ndarray], 
    anchors_img: List[Tuple[float, float]], 
    anchors_field: List[Tuple[float, float]]
) -> Optional[float]:
    """Compute mean reprojection error for homography quality assessment.
    
    Args:
        H: Homography matrix
        anchors_img: Image space anchor points
        anchors_field: Field template space anchor points
        
    Returns:
        Mean reprojection error in pixels, or None if invalid
    """
    if H is None or len(anchors_img) == 0:
        return None
    pts_img = np.array(anchors_img, dtype=np.float32)
    pts_field = np.array(anchors_field, dtype=np.float32)
    proj = _project_points(H, pts_img)
    err = np.linalg.norm(proj - pts_field, axis=1)
    return float(np.mean(err)) if err.size else None


def smooth_homography(
    prev_H: Optional[np.ndarray], 
    new_H: Optional[np.ndarray], 
    alpha: float
) -> Optional[np.ndarray]:
    """Apply exponential moving average smoothing to homography matrices.
    
    Args:
        prev_H: Previous homography matrix
        new_H: New homography matrix
        alpha: Smoothing factor (0=all new, 1=all prev)
        
    Returns:
        Smoothed homography matrix
    """
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
