
import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab
from mask_tab import MaskTab

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "WDS"))

BG      = "#0f1117"
BG2     = "#1a1d27"
BG3     = "#22263a"
ACCENT  = "#00d4ff"
RED     = "#ff4444"
GREEN   = "#00e676"
TEXT    = "#e8eaf6"
SUBTEXT = "#7986cb"
FONT    = ("Segoe UI", 10)
FONT_B  = ("Segoe UI", 10, "bold")
FONT_H  = ("Segoe UI", 14, "bold")
FONT_SM = ("Segoe UI", 9)


class HybridApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OmniParser Enhancer")
        self.configure(bg=BG)
        self.geometry("1280x800")
        self.minsize(900, 600)
        self.resizable(True, True)
        self.bind("<Control-v>", lambda e: self._paste_clipboard())

        # State
        self.input_path   = tk.StringVar(value="")
        self.status_text  = tk.StringVar(value="Ready, select an image to begin.")
        self.stat_omni    = tk.StringVar(value="—")
        self.stat_wds     = tk.StringVar(value="—")
        self.stat_total   = tk.StringVar(value="—")
        self._orig_pil    = None  # original PIL image
        self._out_pil     = None  # result PIL image
        self._zoom        = 1.0
        self._models_loaded = False
        self._last_detections = []
        self._tooltip_id = None
        self._tip_window = None

        self._build_ui()
        self._lazy_load_models()

    def _lazy_load_models(self):
        self._set_status("Loading models in background…", SUBTEXT)
        t = threading.Thread(target=self._load_models_thread, daemon=True)
        t.start()

    def _load_models_thread(self):
        try:
            from util.utils import (
                get_yolo_model, get_caption_model_processor,
                check_ocr_box, get_som_labeled_img,
                get_parsed_content_icon,
            )
            from wehgp import process as wehgp_process

            self._yolo  = get_yolo_model(model_path=os.path.join(ROOT, "weights/icon_detect/model.pt"))
            self._cap   = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path=os.path.join(ROOT, "weights/icon_caption_florence")
            )
            self._wehgp = wehgp_process
            self._check_ocr_box    = check_ocr_box
            self._get_som_labeled  = get_som_labeled_img
            self._get_parsed_content = get_parsed_content_icon
            self._models_loaded    = True
            self.after(0, lambda: self._set_status("Models loaded — select an image.", GREEN))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, lambda: self._set_status(f"Model load error: {e}", RED))

    def _build_ui(self):
        top = tk.Frame(self, bg=BG2, padx=16, pady=10)
        top.pack(fill=tk.X)

        tk.Label(top, text="OmniParser", font=("Segoe UI", 16, "bold"),
                 fg=ACCENT, bg=BG2).pack(side=tk.LEFT)
        tk.Label(top, text=" × WDS", font=("Segoe UI", 16, "bold"),
                 fg=RED, bg=BG2).pack(side=tk.LEFT)
        tk.Label(top, text=" Hybrid Analyzer", font=("Segoe UI", 16),
                 fg=TEXT, bg=BG2).pack(side=tk.LEFT)

        ctrl = tk.Frame(self, bg=BG3, padx=12, pady=8)
        ctrl.pack(fill=tk.X)

        self._btn_pick = self._btn(ctrl, "Select Image", self._pick_file)
        self._btn_pick.pack(side=tk.LEFT, padx=(0,8))

        self._btn_paste = self._btn(ctrl, "Paste", self._paste_clipboard)
        self._btn_paste.pack(side=tk.LEFT, padx=(0,8))

        self._path_lbl = tk.Label(ctrl, textvariable=self.input_path, font=FONT_SM,
                                  fg=SUBTEXT, bg=BG3, anchor="w")
        self._path_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._btn_run = self._btn(ctrl, "Run Analysis", self._run_analysis,
                                  fg=BG, bg=ACCENT, active_bg="#00a8cc")
        self._btn_run.pack(side=tk.RIGHT, padx=(8,0))

        stats = tk.Frame(self, bg=BG2, padx=16, pady=6)
        stats.pack(fill=tk.X)

        for label, var, color in [
            ("OmniParser", self.stat_omni,  ACCENT),
            ("WDS Rescued", self.stat_wds,   RED),
            ("Total",       self.stat_total, GREEN),
        ]:
            f = tk.Frame(stats, bg=BG2, padx=12)
            f.pack(side=tk.LEFT)
            tk.Label(f, text=label, font=FONT_SM, fg=SUBTEXT, bg=BG2).pack()
            tk.Label(f, textvariable=var, font=("Segoe UI", 18, "bold"),
                     fg=color, bg=BG2).pack()

            tk.Label(f, textvariable=var, font=("Segoe UI", 18, "bold"),
                     fg=color, bg=BG2).pack()

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        self.detect_tab = tk.Frame(self._notebook, bg=BG)
        self._notebook.add(self.detect_tab, text="  Detections  ")

        self.mask_tab = MaskTab(self._notebook, enlarge_callback=self._enlarge_image)
        self._notebook.add(self.mask_tab, text="  Laplacian Mask  ")

        self._orig_frame, self._orig_canvas = self._image_panel(self.detect_tab, "Original Image")
        self._orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,6))

        self._out_frame, self._out_canvas = self._image_panel(self.detect_tab, "Hybrid Result")
        self._out_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6,0))

        # Bind clicks
        self._orig_canvas.bind("<Button-1>", lambda e: self._enlarge_image(self._orig_pil))
        self._out_canvas.bind("<Button-1>", lambda e: self._enlarge_image(self._out_pil))
        self._out_canvas.bind("<Motion>", self._on_mouse_move)
        self._out_canvas.bind("<Leave>", lambda e: self._hide_tooltip())

        bot = tk.Frame(self, bg=BG2, padx=12, pady=4)
        bot.pack(fill=tk.X, side=tk.BOTTOM)

        self._status_lbl = tk.Label(bot, textvariable=self.status_text,
                                    font=FONT_SM, fg=SUBTEXT, bg=BG2, anchor="w")
        self._status_lbl.pack(side=tk.LEFT)

        self._progress = ttk.Progressbar(bot, mode="indeterminate", length=160)
        self._progress.pack(side=tk.RIGHT)

        self._style_progressbar()

    def _style_progressbar(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TProgressbar", troughcolor=BG3, background=ACCENT, thickness=4)

        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG3, foreground=SUBTEXT, 
                    padding=[20, 5], borderwidth=0, font=FONT_B)
        s.map("TNotebook.Tab", 
              background=[("selected", BG2), ("active", BG3)],
              foreground=[("selected", ACCENT), ("active", TEXT)])

    def _btn(self, parent, text, cmd, fg=TEXT, bg=BG3, active_bg=BG2):
        b = tk.Button(parent, text=text, command=cmd, font=FONT_B,
                      fg=fg, bg=bg, activeforeground=fg, activebackground=active_bg,
                      bd=0, padx=14, pady=6, cursor="hand2", relief=tk.FLAT)
        b.bind("<Enter>", lambda e: b.config(bg=active_bg))
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    def _image_panel(self, parent, title):
        frame = tk.Frame(parent, bg=BG2, bd=1, relief=tk.FLAT)
        tk.Label(frame, text=title, font=FONT_B, fg=TEXT, bg=BG3,
                 anchor="w", padx=10, pady=5).pack(fill=tk.X)
        canvas = tk.Canvas(frame, bg=BG, highlightthickness=0, cursor="hand2")
        canvas.pack(fill=tk.BOTH, expand=True)
        return frame, canvas

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")]
        )
        if not path:
            return
        self.input_path.set(path)
        self._orig_pil = Image.open(path).convert("RGB")
        self._out_pil  = None
        self._show_image(self._orig_canvas, self._orig_pil)
        self._clear_canvas(self._out_canvas)
        self.mask_tab.clear()
        self.stat_omni.set("—"); self.stat_wds.set("—"); self.stat_total.set("—")
        self._set_status(f"Loaded: {os.path.basename(path)}")

    def _paste_clipboard(self):
        img = ImageGrab.grabclipboard()
        if isinstance(img, Image.Image):
            path = os.path.join(ROOT, "clipboard_temp.png")
            img.convert("RGB").save(path)
            self.input_path.set(path)
            self._orig_pil = img.convert("RGB")
            self._out_pil  = None
            self._show_image(self._orig_canvas, self._orig_pil)
            self._clear_canvas(self._out_canvas)
            self.mask_tab.clear()
            self.stat_omni.set("—"); self.stat_wds.set("—"); self.stat_total.set("—")
            self._set_status("Pasted image from clipboard.")
        elif isinstance(img, list) and img:
            path = img[0]
            if os.path.isfile(path):
                self.input_path.set(path)
                self._orig_pil = Image.open(path).convert("RGB")
                self._out_pil  = None
                self._show_image(self._orig_canvas, self._orig_pil)
                self._clear_canvas(self._out_canvas)
                self.mask_tab.clear()
                self.stat_omni.set("—"); self.stat_wds.set("—"); self.stat_total.set("—")
                self._set_status(f"Loaded from clipboard: {os.path.basename(path)}")
        else:
            messagebox.showinfo("Clipboard", "No image found in clipboard.")

    def _run_analysis(self):
        if not self.input_path.get():
            messagebox.showwarning("No image", "Please select an image first.")
            return
        if not self._models_loaded:
            messagebox.showinfo("Loading", "Models are still loading, please wait.")
            return
        self._btn_run.config(state=tk.DISABLED)
        self._progress.start(12)
        self._set_status("Running WDS + OmniParser…", SUBTEXT)
        t = threading.Thread(target=self._analysis_thread, daemon=True)
        t.start()

    def _analysis_thread(self):
        try:
            import cv2, numpy as np, torch
            from torchvision.transforms import ToPILImage
            from wds_omniparser_bridge import (
                ratio_to_pixel, is_covered, draw_combined,
                BOX_THRESHOLD, IOU_THRESHOLD, IMGSZ,
                _yolo_verify_wds_crops, YOLO_VERIFY_THRESH, YOLO_VERIFY_IMGSZ,
            )

            img_path = self.input_path.get()
            pil_image = Image.open(img_path).convert("RGB")
            bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            W, H = pil_image.size

            self.after(0, lambda: self._set_status("Step 1/3: Running WDS Laplacian detector…", SUBTEXT))
            _, wds_boxes, mask_img = self._wehgp(bgr_image)
            wds_px = [[b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]] for b in wds_boxes]

            self.after(0, lambda: self._set_status("Step 2/3: Running OmniParser YOLO + Florence-2…", SUBTEXT))
            ocr_result, _ = self._check_ocr_box(
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
            _, _, omni_elements = self._get_som_labeled(
                pil_image, self._yolo, BOX_TRESHOLD=BOX_THRESHOLD,
                output_coord_in_ratio=True, ocr_bbox=ocr_bbox,
                draw_bbox_config=dbc, caption_model_processor=self._cap,
                ocr_text=ocr_text, iou_threshold=IOU_THRESHOLD, imgsz=IMGSZ,
            )

            omni_px = [ratio_to_pixel(e["bbox"], W, H) for e in omni_elements if len(e.get("bbox",[])) == 4]

            self.after(0, lambda: self._set_status("Step 3/4: Finding WDS elements missed by OmniParser…", SUBTEXT))
            bgr_np = np.array(pil_image)[:, :, ::-1].copy()
            missed_px = []

            for wbox in wds_px:
                if (wbox[2]-wbox[0]) < 4 or (wbox[3]-wbox[1]) < 4:
                    continue
                if not any(is_covered(wbox, opx) for opx in omni_px):
                    missed_px.append(wbox)

            self.after(0, lambda: self._set_status("Step 3b/4: YOLO crop verification on WDS candidates…", SUBTEXT))
            verified_results = _yolo_verify_wds_crops(
                self._yolo, bgr_np, missed_px,
                conf_thresh=YOLO_VERIFY_THRESH,
                imgsz=YOLO_VERIFY_IMGSZ,
            )
            verified_boxes = [box for box, _ in verified_results]

            self.after(0, lambda: self._set_status("Step 4/4: Captioning YOLO-verified missed SOTs…", SUBTEXT))
            
            missed_elements = []
            if verified_boxes:
                ratio_boxes = []
                for box in verified_boxes:
                    ratio_boxes.append([box[0]/W, box[1]/H, box[2]/W, box[3]/H])
                
                img_rgb_np = np.array(pil_image)
                ratio_tensor = torch.tensor(ratio_boxes)
                
                captions = self._get_parsed_content(ratio_tensor, 0, img_rgb_np, self._cap)
                
                for box, cap in zip(verified_boxes, captions):
                    missed_elements.append({"px_box": box, "content": cap})

            result_pil, detections = draw_combined(pil_image, omni_elements, missed_elements)
            self._out_pil = result_pil
            self._last_detections = detections

            out_path = os.path.splitext(img_path)[0] + "_hybrid.png"
            result_pil.save(out_path)

            n_omni = len(omni_elements)
            n_wds  = len(missed_elements)
            total  = n_omni + n_wds

            def done():
                self.stat_omni.set(str(n_omni))
                self.stat_wds.set(str(n_wds))
                self.stat_total.set(str(total))
                self._show_image(self._out_canvas, result_pil)
                self.mask_tab.update_mask(mask_img, wds_boxes)
                self._progress.stop()
                self._btn_run.config(state=tk.NORMAL)
                self._set_status(f"Done! Saved → {os.path.basename(out_path)}", GREEN)

            self.after(0, done)

        except Exception as e:
            import traceback; traceback.print_exc()
            def err():
                self._progress.stop()
                self._btn_run.config(state=tk.NORMAL)
                self._set_status(f"Error: {e}", RED)
            self.after(0, err)

    def _show_image(self, canvas, pil_img):
        """Fit image inside canvas while preserving aspect ratio."""
        canvas.update_idletasks()
        cw, ch = canvas.winfo_width() or 600, canvas.winfo_height() or 400
        img = pil_img.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        canvas._tk_img = tk_img
        canvas.delete("all")
        x, y = cw // 2, ch // 2
        canvas.create_image(x, y, anchor=tk.CENTER, image=tk_img)

    def _clear_canvas(self, canvas):
        canvas.delete("all")
        canvas._tk_img = None

    def _set_status(self, msg, color=SUBTEXT):
        self.status_text.set(msg)
        self._status_lbl.config(fg=color)

    def _enlarge_image(self, pil_img):
        if not pil_img:
            return

        top = tk.Toplevel(self)
        top.title("Full Size Viewer")
        top.configure(bg=BG)
        try:
            top.state("zoomed")
        except Exception:
            pass

        frame = tk.Frame(top, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(frame, bg=BG, highlightthickness=0)
        vbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add image to canvas
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas._tk_img = tk_img  # Reference
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.config(scrollregion=(0, 0, pil_img.size[0], pil_img.size[1]))

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        top.bind("<Escape>", lambda e: top.destroy())

    def _on_mouse_move(self, event):
        if not self._out_pil or not self._last_detections: return
        
        cw, ch = self._out_canvas.winfo_width(), self._out_canvas.winfo_height()
        iw, ih = self._out_pil.size
        scale = min(cw/iw, ch/ih)
        ox = (cw - iw*scale)/2
        oy = (ch - ih*scale)/2
        
        rx = (event.x - ox) / scale
        ry = (event.y - oy) / scale
        
        target = None
        for d in sorted(self._last_detections, key=lambda x: (x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1])):
            x1, y1, x2, y2 = d['box']
            if x1 <= rx <= x2 and y1 <= ry <= y2:
                target = d
                break
        
        if target:
            self._schedule_tooltip(event.x_root + 15, event.y_root + 10, target['caption'])
        else:
            self._hide_tooltip()

    def _schedule_tooltip(self, x, y, text):
        self._hide_tooltip()
        self._tooltip_id = self.after(300, lambda: self._show_tooltip(x, y, text))

    def _show_tooltip(self, x, y, text):
        if self._tip_window: return
        self._tip_window = tw = tk.Toplevel(self)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"), padx=4, pady=2)
        label.pack()

    def _hide_tooltip(self):
        if self._tooltip_id:
            self.after_cancel(self._tooltip_id)
            self._tooltip_id = None
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None

if __name__ == "__main__":
    app = HybridApp()
    app.mainloop()
