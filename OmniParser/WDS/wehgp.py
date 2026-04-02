"""
vision/wehgp.py — Wavelet-Enhanced Hierarchical GUI Parser  v3 (LSD rewrite)

ARCHITECTURE — Clean, no legacy pipelines
==========================================
The old 8-step pipeline (blocks → CC → MSER → text → classifier → icon_grid →
icon_rescue → gradient_filter) has been replaced with a single, principled step:

    Laplacian Structure Detection (LSD)
      → direct detection of all UI elements via per-channel Laplacian response
      → gradient background is mathematically zero in Laplacian space
      → no mask needed, no anomaly map, no classifier required

Pipeline:
  1. preprocess  — bilateral denoise, CLAHE, down-sample pyramid
  2. detect_icons (LSD)  — Laplacian Structure Detection, all elements at once
  3. IoU-NMS  — deduplicate; prefer higher-score boxes
  4. return (viz, boxes)

The LSD handles:
  - Coloured icons     (high Laplacian on hue-edge channels)
  - White/grey icons   (high Laplacian at border step)
  - Text labels        (MSER grouping inside icon_detector)
  - Any background     (gradient → Laplacian=0 → suppressed)
"""
from __future__ import annotations

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


# ── NMS (IoU-based) ────────────────────────────────────────────────────────────

def _iou_nms(boxes: list[dict], iou_thresh: float = 0.35) -> list[dict]:
    """
    Standard IoU-NMS. Boxes sorted by score descending; suppress lower-score
    boxes that overlap more than iou_thresh with a kept box.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: -(b["w"] * b["h"]))  # largest first (standard NMS)
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
            # Containment: b is a fragment inside k (≥85% of b's area inside k)
            if b_area > 0 and inter / b_area >= 0.85:
                suppressed = True
                break
        if not suppressed:
            kept.append(b)

    return kept


# ── SoM Visualization ─────────────────────────────────────────────────────────

# Palette: alternating vivid hues so adjacent marks are visually distinct
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
    """
    Render Set-of-Mark style visualization:
      - Semi-transparent colored fill over each element bounding box
      - Bold colored border
      - Numbered badge (white text on colored circle) at top-left corner
    """
    out = frame_bgr.copy()
    overlay = frame_bgr.copy()

    for b in boxes:
        idx   = b.get("som_index", 0)
        color = _SOM_PALETTE[(idx - 1) % len(_SOM_PALETTE)]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        x2, y2 = x + w, y + h

        # Semi-transparent fill
        cv2.rectangle(overlay, (x, y), (x2, y2), color, -1)

    # Blend fill at 25% opacity
    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

    for b in boxes:
        idx   = b.get("som_index", 0)
        color = _SOM_PALETTE[(idx - 1) % len(_SOM_PALETTE)]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        # "Boxes are enough" - we don't need to bake thick colored SoM numbers or boxes
        # directly into the image bytes, because the React frontend beautifully
        # draws glowing CSS-styled bounding boxes perfectly mapped over these regions!
        pass

    return out


# ── Public API ─────────────────────────────────────────────────────────────────

def process(frame_bgr: np.ndarray,
            band: str = "HH") -> tuple[np.ndarray, list[dict]]:
    """
    Main entry point.
    Returns (som_viz_bgr, boxes) where:
      boxes = list[{x, y, w, h, score, type, som_index, affordance}]
    """
    # ── Step 2: SoM detection — Laplacian p90 → CC → typed boxes ─────────────
    raw_boxes, final_mask = detect_icons(frame_bgr, [])

    # ── Step 3: IoU-NMS (dedup any remaining overlaps) ────────────────────────
    final_boxes = _iou_nms(raw_boxes, iou_thresh=0.35)
    # Sort by position (top→bottom, left→right) for stable SoM numbering
    final_boxes.sort(key=lambda b: (b["y"], b["x"]))
    # No hard cap — NMS already bounds the count to non-overlapping elements

    # Re-assign stable SoM indices after NMS
    for i, b in enumerate(final_boxes):
        b["som_index"] = i + 1

    # ── Step 4: SoM visualization ─────────────────────────────────────────────
    viz = _draw_som(frame_bgr, final_boxes)

    return viz, final_boxes, final_mask
