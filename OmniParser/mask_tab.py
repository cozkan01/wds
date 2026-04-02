import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

class MaskTab(tk.Frame):
    def __init__(self, parent, enlarge_callback=None):
        super().__init__(parent, bg="#0f1117")
        self.enlarge_callback = enlarge_callback
        
        # UI
        tk.Label(self, text="Laplacian Binary Mask — Visual Correlation", 
                 font=("Segoe UI", 10, "bold"), fg="#e8eaf6", bg="#1a1d27", 
                 anchor="w", padx=10, pady=5).pack(fill=tk.X)
                 
        self.canvas = tk.Canvas(self, bg="#000000", highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self._on_click)
        
        self.mask_pil = None
        self._tk_img = None

    def update_mask(self, mask_np, boxes):
        """
        mask_np: uint8 binary mask (0 or 255)
        boxes: list of {'x','y','w','h'}
        """
        # Create a colorized visualization
        # Base: Dark bluish
        h, w = mask_np.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Edge pixels: Cyan
        vis[mask_np > 0] = [0, 212, 255] 
        
        # Draw SOT boxes: Red
        for b in boxes:
            x, y, w_box, h_box = b['x'], b['y'], b['w'], b['h']
            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (255, 68, 68), 2)
            
        self.mask_pil = Image.fromarray(vis)
        self._show_on_canvas()

    def clear(self):
        self.canvas.delete("all")
        self.mask_pil = None
        self._tk_img = None

    def _show_on_canvas(self):
        if not self.mask_pil: return
        
        self.canvas.update_idletasks()
        cw, ch = self.canvas.winfo_width() or 800, self.canvas.winfo_height() or 600
        
        img = self.mask_pil.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)
        
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._tk_img)

    def _on_click(self, event):
        if self.mask_pil and self.enlarge_callback:
            self.enlarge_callback(self.mask_pil)
