"""Generate a simple canonical football field template image.

This script creates a top-down green field with sidelines, yard lines every 5 yards,
numbers every 10 yards, and approximate hash marks. It saves to assets/field_template.png.

Customize dimensions and layout as needed. The pipeline uses the template size at runtime.
"""

from pathlib import Path
import cv2
import numpy as np


def draw_field(w: int = 1800, h: int = 800, out: Path = Path("assets/field_template.png")):
    out.parent.mkdir(parents=True, exist_ok=True)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 120, 30)  # grass green (BGR)

    # Field boundaries (approximate 120x53.33 yards mapped to width x height)
    pad_x = int(0.02 * w)
    pad_y = int(0.08 * h)
    x0, x1 = pad_x, w - pad_x
    y0, y1 = pad_y, h - pad_y

    # Draw sidelines and end lines
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 4)

    def yard_to_x(yards: float) -> int:
        # Map 0..100 to x0..x1 (end zones excluded for simplicity)
        return int(x0 + (yards / 100.0) * (x1 - x0))

    # Yard lines every 5 yards
    for yd in range(0, 101, 5):
        x = yard_to_x(yd)
        thickness = 3 if yd % 10 == 0 else 1
        cv2.line(img, (x, y0), (x, y1), (255, 255, 255), thickness)

    # Hash marks: two rows near center (approximate)
    hash_top = int(y0 + 0.35 * (y1 - y0))
    hash_bot = int(y0 + 0.65 * (y1 - y0))
    for yd in range(0, 101):
        x = yard_to_x(yd)
        cv2.line(img, (x, hash_top - 6), (x, hash_top + 6), (255, 255, 255), 2)
        cv2.line(img, (x, hash_bot - 6), (x, hash_bot + 6), (255, 255, 255), 2)

    # Numbers every 10 yards near top/bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, yd in enumerate(range(10, 100, 10)):
        label = str(yd if yd <= 50 else 100 - yd)
        x = yard_to_x(yd)
        # Slight offset so numbers don't overlap lines
        cv2.putText(img, label, (x - 18, y0 + 40), font, 1.0, (255, 255, 255), 2)
        cv2.putText(img, label, (x - 18, y1 - 20), font, 1.0, (255, 255, 255), 2)

    cv2.imwrite(str(out), img)
    print(f"Saved field template to {out} ({w}x{h})")


if __name__ == "__main__":
    draw_field()
