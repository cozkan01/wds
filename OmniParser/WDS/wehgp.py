
from __future__ import annotation
import cv2
import numpy as np

from icon_detector import detect_icons


_TYPE_PALETTE = {
    "text":      (0,   255, 136),
    "label":     (68,  255, 170),
    "button":    (0,   153, 255),
    "icon":      (255, 212,   0),
    "container": (204, 136, 255),
    "element":   (180, 180, 180),
}
def _iou_nms(boxes: list[dict], iou_thresh: float = 0.35) -> list[dict]:
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: -(b["w"] * b["h"]))
    kept: list[dict] = []

    for b in boxes:
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = bx1 + b["w"], by1 + b["h"]
        b_area = b["w"] * b["h"]
        suppressed = False
        for k in kept:
            kx1, ky1 = k["x"], k["y"]
            kx2, ky2 = kx1 + k["w"], ky1 + k["h"]
            ix = max(0, min(bx2, kx2) - max(bx1, kx1))
            iy = max(0, min(by2, ky2) - max(by1, ky1))
            inter = ix * iy
            union = b_area + k["w"] * k["h"] - inter
            iou   = inter / (union + 1e-8)
            # Standard IoU overlap
            if iou > iou_thresh:
                suppressed = True
                break
            if b_area > 0 and inter / b_area >= 0.85:
                suppressed = True
                break
        if not suppressed:
            kept.append(b)

    return kept



_SOM_PALETTE = [
    (255, 80,  80),   # red
    (80,  200, 255),  # cyan
    (255, 200, 60),   # yellow
    (130, 255, 100),  # green
    (200, 100, 255),  # purple
    (255, 140, 40),   # orange
    (60,  180, 255),  # sky
    (255, 80,  200),  # pink
]


def _draw_som(frame_bgr: np.ndarray, boxes: list[dict]) -> np.ndarray:
    out = frame_bgr.copy()
    overlay = frame_bgr.copy()

    for b in boxes:
        idx   = b.get("som_index", 0)
        color = _SOM_PALETTE[(idx - 1) % len(_SOM_PALETTE)]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        x2, y2 = x + w, y + h

        cv2.rectangle(overlay, (x, y), (x2, y2), color, -1)

    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

    for b in boxes:
        idx   = b.get("som_index", 0)
        color = _SOM_PALETTE[(idx - 1) % len(_SOM_PALETTE)]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        pass

    return out



def process(frame_bgr: np.ndarray,
            band: str = "HH") -> tuple[np.ndarray, list[dict]]:
    """
    Main entry point.
    Returns (som_viz_bgr, boxes) where:
      boxes = list[{x, y, w, h, score, type, som_index, affordance}]
    """
    raw_boxes, final_mask = detect_icons(frame_bgr, [])

    final_boxes = _iou_nms(raw_boxes, iou_thresh=0.35)
    final_boxes.sort(key=lambda b: (b["y"], b["x"]))

    for i, b in enumerate(final_boxes):
        b["som_index"] = i + 1

    viz = _draw_som(frame_bgr, final_boxes)

    return viz, final_boxes, final_mask
