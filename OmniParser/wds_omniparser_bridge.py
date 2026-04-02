"""
(Laplacian SOT detector)
"""

import sys
import os
import io
import base64
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "WDS"))
from wehgp import process as wehgp_process

from util.utils import (
    get_yolo_model,
    get_caption_model_processor,
    get_parsed_content_icon,
    check_ocr_box,
    get_som_labeled_img,
)

# ──────────────────────────────────────────────────────────────────────────────
IMAGE_PATH       = "test1.png"
YOLO_PATH        = "weights/icon_detect/model.pt"
CAPTION_PATH     = "weights/icon_caption_florence"
OUTPUT_PATH      = "output_test1.png"
IOU_MISS_THRESH  = 0.10
BOX_THRESHOLD    = 0.05
IOU_THRESHOLD    = 0.10
IMGSZ            = 640
YOLO_VERIFY_THRESH = 0.15
YOLO_VERIFY_IMGSZ  = 256   
CROP_PAD_X         = 0.20
CROP_PAD_Y_ABOVE   = 0.20  
CROP_PAD_Y_BELOW   = 0.30
WDS_BETTER_AREA_RATIO = 2.0

# ──────────────────────────────────────────────────────────────────────────────
def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def is_covered(wds_box, omni_box, iou_thresh=0.10, containment_thresh=0.70,
               wds_better_ratio=WDS_BETTER_AREA_RATIO):
    ix1 = max(wds_box[0], omni_box[0]); iy1 = max(wds_box[1], omni_box[1])
    ix2 = min(wds_box[2], omni_box[2]); iy2 = min(wds_box[3], omni_box[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return False
    area_wds  = max(1, (wds_box[2]-wds_box[0])  * (wds_box[3]-wds_box[1]))
    area_omni = max(1, (omni_box[2]-omni_box[0]) * (omni_box[3]-omni_box[1]))

    union = area_wds + area_omni - inter
    iou = inter / union
    
    if iou >= 0.15:
        return True

    containment = inter / area_wds
    if containment >= 0.70:
        return True

    return False



def _expand_box(
    box: list[int], W: int, H: int,
    pad_x: float = CROP_PAD_X,
    pad_y_above: float = CROP_PAD_Y_ABOVE,
    pad_y_below: float = CROP_PAD_Y_BELOW,
) -> list[int]:
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    x1p = max(0,     int(x1 - bw * pad_x))
    y1p = max(0,     int(y1 - bh * pad_y_above))
    x2p = min(W - 1, int(x2 + bw * pad_x))
    y2p = min(H - 1, int(y2 + bh * pad_y_below))
    return [x1p, y1p, x2p, y2p]


def _yolo_verify_wds_crops(
    yolo_model,
    bgr_image: np.ndarray,
    candidate_px: list[list[int]],
    conf_thresh: float = YOLO_VERIFY_THRESH,
    imgsz: int = YOLO_VERIFY_IMGSZ,
) -> list[tuple[list[int], float]]:
    from PIL import Image as PILImage
    H_img, W_img = bgr_image.shape[:2]

    pil_crops: list[PILImage.Image] = []
    valid_boxes: list[list[int]] = []

    for box in candidate_px:
        x1, y1, x2, y2 = box
        x1c, y1c, x2c, y2c = _expand_box(box, W_img, H_img)
        if (x2c - x1c) < 4 or (y2c - y1c) < 4:
            continue
        crop_bgr = bgr_image[y1c:y2c, x1c:x2c]
        h, w = crop_bgr.shape[:2]
        side = max(h, w)
        padded = np.zeros((side, side, 3), dtype=np.uint8)
        padded[:h, :w] = crop_bgr
        crop_resized = cv2.resize(padded, (imgsz, imgsz))
        pil_crops.append(PILImage.fromarray(crop_resized[:, :, ::-1]))  # BGR→RGB
        valid_boxes.append([x1c, y1c, x2c, y2c])

    if not pil_crops:
        return []

    results = yolo_model.predict(
        source=pil_crops,
        conf=conf_thresh,
        imgsz=imgsz,
        iou=0.5,
        verbose=False,
    )

    verified: list[tuple[list[int], float]] = []
    for box, result in zip(valid_boxes, results):
        if len(result.boxes) == 0:
            continue
            
        conf_scores = result.boxes.conf
        best_idx = int(conf_scores.argmax())
        best_conf = float(conf_scores[best_idx])
        
        if best_conf >= conf_thresh:
            det_box = result.boxes.xyxy[best_idx].cpu().numpy()
            cx, cy = imgsz / 2, imgsz / 2
            core = imgsz * 0.25
            x1_c, y1_c, x2_c, y2_c = cx - core, cy - core, cx + core, cy + core
            
            ix1, iy1 = max(det_box[0], x1_c), max(det_box[1], y1_c)
            ix2, iy2 = min(det_box[2], x2_c), min(det_box[3], y2_c)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            
            if inter > 0:
                verified.append((box, best_conf))

    return verified


def ratio_to_pixel(box_ratio, W, H):
    """Convert OmniParser ratio bbox [x1r,y1r,x2r,y2r] to pixel [x1,y1,x2,y2]."""
    return [
        int(box_ratio[0] * W), int(box_ratio[1] * H),
        int(box_ratio[2] * W), int(box_ratio[3] * H),
    ]


def _draw_badge(draw, x, y, label, bg, font):
    """Draw a compact pill-shaped badge at (x, y)."""
    tw = max(len(label) * 7 + 8, 20)
    th = 16
    draw.rounded_rectangle([x, y - th, x + tw, y], radius=4, fill=bg)
    draw.text((x + 4, y - th + 2), label, fill=(255, 255, 255), font=font)


def draw_combined(image_pil, omni_elements, missed_elements):
    """
    Draw all detections on the image:
      - OmniParser hits  → cyan boxes with small index badge
      - WDS-rescued SOTs → red boxes with small 'W#' badge
    Returns (annotated PIL Image, detections list) where each entry is
    {'box': [x1,y1,x2,y2], 'label': str, 'caption': str, 'source': 'omni'|'wds'}
    so the GUI can build a hover tooltip lookup.
    """
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_sm = ImageFont.load_default()

    W, H = img.size
    detections = []

    for i, elem in enumerate(omni_elements):
        bbox = elem.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = ratio_to_pixel(bbox, W, H)
        idx = i + 1
        draw.rectangle([x1, y1, x2, y2], outline=(0, 210, 255, 230), width=2)
        _draw_badge(draw, x1, y1, str(idx), (0, 140, 180, 220), font_sm)
        detections.append({
            "box": [x1, y1, x2, y2], "label": str(idx),
            "caption": str(elem.get("content", "")), "source": "omni",
        })

    for i, elem in enumerate(missed_elements):
        x1, y1, x2, y2 = elem["px_box"]
        idx = i + 1
        draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60, 255), width=2)
        _draw_badge(draw, x1, y1, f"W{idx}", (200, 30, 30, 220), font_sm)
        detections.append({
            "box": [x1, y1, x2, y2], "label": f"W{idx}",
            "caption": str(elem.get("content", "?")), "source": "wds",
        })

    return img, detections


# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"[bridge] Loading image: {IMAGE_PATH}")
    pil_image = Image.open(IMAGE_PATH).convert("RGB")
    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    W, H = pil_image.size

    print("[bridge] Running WDS Laplacian SOT detector...")
    _, wds_boxes, _ = wehgp_process(bgr_image)
    print(f"[bridge]   WDS found {len(wds_boxes)} SOTs")

    wds_px = []
    for b in wds_boxes:
        x1, y1 = b["x"], b["y"]
        x2, y2 = x1 + b["w"], y1 + b["h"]
        wds_px.append([x1, y1, x2, y2])

    print("[bridge] Loading OmniParser models...")
    yolo_model = get_yolo_model(model_path=YOLO_PATH)
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", model_name_or_path=CAPTION_PATH
    )

    print("[bridge] Running OmniParser OCR...")
    ocr_result, _ = check_ocr_box(
        pil_image, display_img=False, output_bb_format="xyxy",
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=False,
    )
    ocr_text, ocr_bbox = ocr_result

    print("[bridge] Running OmniParser YOLO + Florence-2...")
    box_overlay_ratio = W / 3200
    draw_bbox_config = {
        "text_scale":     0.8  * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding":   max(int(3 * box_overlay_ratio), 1),
        "thickness":      max(int(3 * box_overlay_ratio), 1),
    }
    _, label_coords, omni_elements = get_som_labeled_img(
        pil_image, yolo_model,
        BOX_TRESHOLD=BOX_THRESHOLD,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=ocr_text,
        iou_threshold=IOU_THRESHOLD,
        imgsz=IMGSZ,
    )
    print(f"[bridge]   OmniParser found {len(omni_elements)} elements")

    omni_px = []
    for elem in omni_elements:
        bbox = elem.get("bbox", [])
        if len(bbox) == 4:
            omni_px.append(ratio_to_pixel(bbox, W, H))
        else:
            omni_px.append(None)

    print("[bridge] Comparing WDS SOTs vs OmniParser boxes...")
    missed_px = []
    for wbox in wds_px:
        if (wbox[2] - wbox[0]) < 4 or (wbox[3] - wbox[1]) < 4:
            continue
        covered = any(
            is_covered(wbox, opx)
            for opx in omni_px
            if opx is not None
        )
        if not covered:
            missed_px.append(wbox)

    print(f"[bridge]   WDS SOTs missed by OmniParser: {len(missed_px)}")

    print("[bridge] Running YOLO crop verification on WDS candidates...")
    verified_results = _yolo_verify_wds_crops(
        yolo_model, bgr_np := np.array(pil_image)[:, :, ::-1].copy(),
        missed_px,
        conf_thresh=YOLO_VERIFY_THRESH,
        imgsz=YOLO_VERIFY_IMGSZ,
    )
    verified_boxes = [box for box, _ in verified_results]
    rejected = len(missed_px) - len(verified_boxes)
    print(f"[bridge]   YOLO verified: {len(verified_boxes)} | Rejected as non-UI: {rejected}")

    bgr_np = np.array(pil_image)[:, :, ::-1].copy()  # RGB → BGR
    print("[bridge] Captioning missed SOTs via Florence-2...")

    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()

    cropped_pils = []
    valid_missed = []
    for box in verified_boxes:
        x1, y1, x2, y2 = box
        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(W, x2); y2c = min(H, y2)
        if (x2c - x1c) < 4 or (y2c - y1c) < 4:
            continue
        crop_bgr = bgr_np[y1c:y2c, x1c:x2c]
        crop_resized = cv2.resize(crop_bgr, (64, 64))
        cropped_pils.append(to_pil(crop_resized[:, :, ::-1]))
        valid_missed.append([x1c, y1c, x2c, y2c])

    missed_elements = []
    if cropped_pils:
        model = caption_model_processor["model"]
        processor = caption_model_processor["processor"]
        device = model.device
        prompt = "<CAPTION>"
        batch_size = 64

        captions = []
        for i in range(0, len(cropped_pils), batch_size):
            batch = cropped_pils[i:i+batch_size]
            if model.device.type == "cuda":
                inputs = processor(
                    images=batch, text=[prompt]*len(batch),
                    return_tensors="pt", do_resize=False
                ).to(device=device, dtype=torch.float16)
            else:
                inputs = processor(
                    images=batch, text=[prompt]*len(batch),
                    return_tensors="pt"
                ).to(device=device)
            with torch.inference_mode():
                ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=20, num_beams=1, do_sample=False
                )
            texts = processor.batch_decode(ids, skip_special_tokens=True)
            captions.extend([t.strip() for t in texts])

        for box, cap in zip(valid_missed, captions):
            missed_elements.append({"px_box": box, "content": cap})
        print(f"[bridge]   Captioned {len(missed_elements)} rescued SOTs")
    else:
        print("[bridge]   No valid missed SOTs to caption.")

    print("[bridge] Rendering combined output image...")
    final_img, _ = draw_combined(pil_image, omni_elements, missed_elements)
    final_img.save(OUTPUT_PATH)
    print(f"[bridge] Done! Saved → {OUTPUT_PATH}")
    print(f"\n  OmniParser elements : {len(omni_elements)}")
    print(f"  WDS rescued SOTs    : {len(missed_elements)}")
    print(f"  Total detections    : {len(omni_elements) + len(missed_elements)}")


if __name__ == "__main__":
    main()
