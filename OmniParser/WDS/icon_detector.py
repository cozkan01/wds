import cv2
import numpy as np

_MORPH_THRESH = 30
_MIN_D = 4
_MAX_D = 180
_OVERSIZE_D = 193
_MIN_AREA = _MIN_D * _MIN_D // 4
_ENTROPY_THRESH = 1.6
_MIN_VAR = 8.0
_FILL_THRESH = 0.15

def _edge_map(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denoised = cv2.bilateralFilter(bgr, 5, 50, 50)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    channels = cv2.split(hsv)
    
    lap_fused = np.zeros(bgr.shape[:2], dtype=np.float32)
    for c in channels:
        lap = cv2.Laplacian(c, cv2.CV_32F, ksize=3)
        cv2.max(lap_fused, np.abs(lap), lap_fused)
        
    score_np = cv2.normalize(lap_fused, None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    _, binary = cv2.threshold(lap_fused, 60, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    binary = cv2.medianBlur(binary, 3)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    gray_f = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_sq = cv2.blur(gray_f ** 2, (7, 7))
    sq_mean = cv2.blur(gray_f, (7, 7)) ** 2
    variance = mean_sq - sq_mean
    
    mask[(variance < 80.0) & (score_np < 0.35)] = 0
    
    return score_np, mask

def _angle_entropy(gx_full: np.ndarray, gy_full: np.ndarray, cc_mask: np.ndarray) -> float:
    gx = gx_full[cc_mask]
    gy = gy_full[cc_mask]
    mag = np.sqrt(gx**2 + gy**2)
    valid = mag > 5.0
    if not np.any(valid):
        return 0.0
    gx, gy, mag = gx[valid], gy[valid], mag[valid]
    angles = np.arctan2(gy, gx) * 180 / np.pi
    angles[angles < 0] += 180
    hist, _ = np.histogram(angles, bins=8, range=(0, 180))
    p = hist / (hist.sum() + 1e-8)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def _try_accept(x: int, y: int, w: int, h: int, area: int, mask: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> tuple[float, bool]:
    if w < _MIN_D or h < _MIN_D:
        return 0.0, False
    if w > _MAX_D or h > _MAX_D:
        return 0.0, False
    
    ar = w / h
    if ar > 5.5 or ar < 0.18:
        return 0.0, False

    fill = area / (w * h)
    if fill < 0.20:
        return 0.0, False
        
    ent = _angle_entropy(gx, gy, mask)
    if ent < _ENTROPY_THRESH:
        return 0.0, False
        
    return fill, True

def detect_icons(bgr: np.ndarray, existing_boxes=None) -> tuple[list[dict], np.ndarray]:
    if existing_boxes is None:
        existing_boxes = []
        
    score_np, mask = _edge_map(bgr)
    
    # Calculate Gx/Gy once for entropy check
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    boxes = []
    labels_keep = np.zeros_like(mask)
    
    oversize_d = _OVERSIZE_D
    
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        
        if w > oversize_d or h > oversize_d:
            # Recursive split...
            cc_mask = (labels == i).astype(np.uint8) * 255
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(cc_mask, k, iterations=1)
            nn, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
            
            k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            for j in range(1, nn):
                seed = (sub_labels == j).astype(np.uint8) * 255
                grown = cv2.dilate(seed, k_d, iterations=1)
                grown = cv2.bitwise_and(grown, cc_mask)
                rr, cc = np.where(grown > 0)
                if len(rr) == 0:
                    continue
                    
                sbx, sby = int(cc.min()), int(rr.min())
                sbw = int(cc.max()) - sbx + 1
                sbh = int(rr.max()) - sby + 1
                
                if sbh > oversize_d or sbw > oversize_d:
                    # Second pass...
                    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    eroded2 = cv2.erode(grown, k2, iterations=2)
                    n2, lbl2, _, _ = cv2.connectedComponentsWithStats(eroded2, connectivity=8)
                    k_d2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    for s2 in range(1, n2):
                        m2 = (lbl2 == s2).astype(np.uint8) * 255
                        g2 = cv2.dilate(m2, k_d2, iterations=1)
                        g2 = cv2.bitwise_and(g2, grown)
                        rr2, cc2 = np.where(g2 > 0)
                        if len(rr2) == 0: continue
                        s2x, s2y = int(cc2.min()), int(rr2.min())
                        s2w, s2h = int(cc2.max())-s2x+1, int(rr2.max())-s2y+1
                        a2 = int((g2 > 0).sum())
                        _, ok2 = _try_accept(s2x, s2y, s2w, s2h, a2, g2.astype(bool), gx, gy)
                        if ok2:
                            boxes.append({"x": s2x, "y": s2y, "w": s2w, "h": s2h, "score": 1.0, "type": "icon"})
                            labels_keep[g2 > 0] = 255
                    continue

                sba = int((grown > 0).sum())
                _, ok = _try_accept(sbx, sby, sbw, sbh, sba, grown.astype(bool), gx, gy)
                if ok:
                    boxes.append({"x": sbx, "y": sby, "w": sbw, "h": sbh, "score": 1.0, "type": "icon"})
                    labels_keep[grown > 0] = 255
            continue
            
        cc_mask_bool = (labels == i)
        _, ok = _try_accept(x, y, w, h, area, cc_mask_bool, gx, gy)
        if ok:
            btype = "label" if w > h * 2.5 else "icon"
            boxes.append({"x": x, "y": y, "w": w, "h": h, "score": 1.0, "type": btype})
            labels_keep[cc_mask_bool] = 255

    return boxes, labels_keep
