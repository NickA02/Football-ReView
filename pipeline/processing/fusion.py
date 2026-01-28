"""Detection fusion and anchor generation."""
from typing import List, Dict, Tuple, Optional
import numpy as np
from ..utils.mappings import cls_to_yards


def fuse_detections(hash_dets: List[Dict], number_dets: List[Dict], template_size: Tuple[int, int]) -> Dict:
    """Fuse detections prioritizing hash lines as anchors.

    Strategy:
      - Parse hash keypoints as two points per vertical yard line: (top_hash_intersection, bottom_hash_intersection).
      - Find nearest hash line to each detected yard number and assign its absolute yard value (10,20,...,50).
      - Propagate ±5 yards to neighboring hash lines by x-order to get at least two labeled lines.
      - Build anchors: for each labeled line, add two correspondences (top/bottom hash rows) in image vs template.

    Args:
        hash_dets: List of hash mark detections
        number_dets: List of yard number detections
        template_size: (width, height) of field template

    Returns:
        Dict with:
        - yard_value: Optional[int] (best yard value from numbers)
        - anchors_img: List[(x,y)]
        - anchors_field: List[(X,Y)] matching template pixels
        - labeled_lines: List of all labeled lines for temporal cache
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
        yard_value = max(num_obs, key=lambda n: n["conf"])["yard"]

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
