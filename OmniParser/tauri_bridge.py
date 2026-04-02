import sys
import os
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from PIL import Image, ImageGrab

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "WDS"))

class AnalyzerState:
    def __init__(self):
        self.models_loaded = False
        self.yolo = None
        self.cap = None
        self.wehgp = None
        self.check_ocr_box = None
        self.get_som_labeled = None
        self.get_parsed_content = None

    def load(self):
        print("Loading models in background...")
        from util.utils import (
            get_yolo_model, get_caption_model_processor,
            check_ocr_box, get_som_labeled_img,
            get_parsed_content_icon,
        )
        from wehgp import process as wehgp_process

        self.yolo = get_yolo_model(model_path=os.path.join(ROOT, "weights/icon_detect/model.pt"))
        self.cap = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path=os.path.join(ROOT, "weights/icon_caption_florence")
        )
        self.wehgp = wehgp_process
        self.check_ocr_box = check_ocr_box
        self.get_som_labeled = get_som_labeled_img
        self.get_parsed_content = get_parsed_content_icon
        self.models_loaded = True
        print("Models loaded successfully.")

def _try_win32_paste(out_path: str):
    """Fallback: read CF_DIB from win32clipboard and save as PNG."""
    try:
        import win32clipboard, io
        win32clipboard.OpenClipboard()
        try:
            if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_DIB):
                data = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)
                # CF_DIB is a BITMAPINFOHEADER + pixel data; wrap with BMP header
                import struct
                header_size = struct.unpack_from('<I', data, 0)[0]
                width       = struct.unpack_from('<i', data, 4)[0]
                height      = struct.unpack_from('<i', data, 8)[0]
                bpp         = struct.unpack_from('<H', data, 14)[0]
                # Build minimal BMP file header
                file_size = 14 + len(data)
                bmp_header = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, 14 + header_size)
                bmp_data = bmp_header + data
                img = Image.open(io.BytesIO(bmp_data))
                img.convert("RGB").save(out_path)
                return out_path
        finally:
            win32clipboard.CloseClipboard()
    except Exception:
        pass
    return None

STATE = AnalyzerState()


class RequestHandler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        # Add CORS headers so Tauri frontend can call it directly if it wanted to! (but we'll call from Rust)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        body = json.loads(post_data.decode('utf-8')) if post_data else {}
        
        path = urlparse(self.path).path

        if path == "/load_models":
            if not STATE.models_loaded:
                try:
                    # Load inline for now
                    STATE.load()
                    self.send_json({"status": "ok"})
                except Exception as e:
                    self.send_json({"error": str(e)}, 500)
            else:
                self.send_json({"status": "already_loaded"})
                
        elif path == "/paste_clipboard":
            # Try PIL ImageGrab first, then win32clipboard as fallback
            img = None
            clipboard_file = None
            try:
                img = ImageGrab.grabclipboard()
            except Exception:
                pass

            if isinstance(img, Image.Image):
                out_path = os.path.join(ROOT, "clipboard_temp.png")
                img.convert("RGB").save(out_path)
                self.send_json({"path": out_path})
            elif isinstance(img, list) and img:
                file_path = img[0]
                if os.path.isfile(file_path):
                    self.send_json({"path": file_path})
                else:
                    self.send_json({"error": "Clipboard file not found"}, 400)
            else:
                # fallback: try win32clipboard for CF_DIB
                saved = _try_win32_paste(os.path.join(ROOT, "clipboard_temp.png"))
                if saved:
                    self.send_json({"path": saved})
                else:
                    self.send_json({"error": "No image in clipboard"}, 400)
                
        elif path == "/analyze":
            if not STATE.models_loaded:
                self.send_json({"error": "Models not loaded"}, 400)
                return
            
            img_path = body.get("imagePath")
            if not img_path:
                self.send_json({"error": "Missing imagePath"}, 400)
                return
            # Normalise path separators
            img_path = img_path.replace("/", os.sep).replace("\\", os.sep)
            if not os.path.exists(img_path):
                self.send_json({"error": f"File not found: {img_path}"}, 400)
                return

            try:
                import cv2, numpy as np, torch
                from wds_omniparser_bridge import (
                    ratio_to_pixel, is_covered, draw_combined,
                    BOX_THRESHOLD, IOU_THRESHOLD, IMGSZ,
                    _yolo_verify_wds_crops, YOLO_VERIFY_THRESH, YOLO_VERIFY_IMGSZ,
                )

                pil_image = Image.open(img_path).convert("RGB")
                bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                W, H = pil_image.size

                # 1. WDS
                _, wds_boxes, mask_img = STATE.wehgp(bgr_image)
                wds_px = [[b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]] for b in wds_boxes]

                # 2. OmniParser
                ocr_result, _ = STATE.check_ocr_box(
                    pil_image, display_img=False, output_bb_format="xyxy",
                    easyocr_args={"paragraph": False, "text_threshold": 0.9},
                    use_paddleocr=False,
                )
                ocr_text, ocr_bbox = ocr_result
                bor = W / 3200
                dbc = {
                    "text_scale": 0.8*bor, "text_thickness": max(int(2*bor),1),
                    "text_padding": max(int(3*bor),1), "thickness": max(int(3*bor),1),
                }
                _, _, omni_elements = STATE.get_som_labeled(
                    pil_image, STATE.yolo, BOX_TRESHOLD=BOX_THRESHOLD,
                    output_coord_in_ratio=True, ocr_bbox=ocr_bbox,
                    draw_bbox_config=dbc, caption_model_processor=STATE.cap,
                    ocr_text=ocr_text, iou_threshold=IOU_THRESHOLD, imgsz=IMGSZ,
                )

                omni_px = [ratio_to_pixel(e["bbox"], W, H) for e in omni_elements if len(e.get("bbox",[])) == 4]

                # 3. WDS misses
                bgr_np = np.array(pil_image)[:, :, ::-1].copy()
                missed_px = []
                for wbox in wds_px:
                    if (wbox[2]-wbox[0]) < 4 or (wbox[3]-wbox[1]) < 4: continue
                    if not any(is_covered(wbox, opx) for opx in omni_px):
                        missed_px.append(wbox)

                # 3b. Verify
                verified_results = _yolo_verify_wds_crops(
                    STATE.yolo, bgr_np, missed_px,
                    conf_thresh=YOLO_VERIFY_THRESH,
                    imgsz=YOLO_VERIFY_IMGSZ,
                )
                verified_boxes = [box for box, _ in verified_results]

                missed_elements = []
                if verified_boxes:
                    ratio_boxes = [[box[0]/W, box[1]/H, box[2]/W, box[3]/H] for box in verified_boxes]
                    img_rgb_np = np.array(pil_image)
                    ratio_tensor = torch.tensor(ratio_boxes)
                    captions = STATE.get_parsed_content(ratio_tensor, 0, img_rgb_np, STATE.cap)
                    for box, cap in zip(verified_boxes, captions):
                        missed_elements.append({"px_box": box, "content": cap})

                # Draw
                result_pil, _ = draw_combined(pil_image, omni_elements, missed_elements)
                
                out_path = os.path.splitext(img_path)[0] + "_hybrid.png"
                result_pil.save(out_path)

                # Draw boxes on mask image
                mask_vis = mask_img.copy()
                if len(mask_vis.shape) == 2:
                    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                # Draw all WDS boxes in red
                for box in wds_px:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(mask_vis, (x1, y1), (x2, y2), (0, 80, 220), 1)
                # Draw verified (missed) boxes with brighter colour
                for box in verified_boxes:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(mask_vis, (x1, y1), (x2, y2), (0, 60, 255), 2)
                # Draw omni boxes in cyan
                for opx in omni_px:
                    x1, y1, x2, y2 = [int(v) for v in opx]
                    cv2.rectangle(mask_vis, (x1, y1), (x2, y2), (200, 160, 0), 1)

                mask_path = os.path.join(ROOT, "mask_temp.png")
                cv2.imwrite(mask_path, mask_vis)

                resp = {
                    "omni_count": len(omni_elements),
                    "wds_count": len(missed_elements),
                    "total_count": len(omni_elements) + len(missed_elements),
                    "output_path": out_path,
                    "mask_path": mask_path
                }
                self.send_json(resp)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json({"error": str(e)}, 500)
        else:
            self.send_json({"error": "not found"}, 404)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/image":
            img_path = parse_qs(urlparse(self.path).query).get('path', [None])[0]
            if img_path and os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as f:
                        data = f.read()
                    self.send_response(200)
                    ext = os.path.splitext(img_path)[1].lower()
                    mime = "image/png" if ext == ".png" else "image/jpeg" if ext in ['.jpg', '.jpeg'] else "application/octet-stream"
                    self.send_header('Content-Type', mime)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data)
                except Exception as e:
                    self.send_json({"error": str(e)}, 500)
            else:
                self.send_json({"error": "File not found"}, 404)
        else:
            self.send_json({"error": "not found"}, 404)

def run_server():
    server_address = ('127.0.0.1', 8991)
    httpd = HTTPServer(server_address, RequestHandler)
    print("Bridge server running on port 8991")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
