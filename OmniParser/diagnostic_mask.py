import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, 'WDS')
from wehgp import process as wehgp_process

TEST_IMAGE = "clipboard_temp.png"
if not os.path.exists(TEST_IMAGE):
    TEST_IMAGE = "test1.png"

def show_mask():
    if not os.path.exists(TEST_IMAGE):
        print(f"Test image {TEST_IMAGE} not found.")
        return

    print(f"--- Running Diagnostic Mask Visualization on {TEST_IMAGE} ---")
    bgr = cv2.imread(TEST_IMAGE)
    if bgr is None:
        print("Could not load image.")
        return

    _, boxes, mask = wehgp_process(bgr)
    
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_vis[mask > 0] = [0, 255, 255]
    
    for i, b in enumerate(boxes):
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        cv2.rectangle(mask_vis, (x, y), (x+w, y+h), (255, 80, 80), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask_vis, f"#{i+1}", (x, y-10), font, 0.4, (255, 80, 80), 1)

    output_path = "output_wds_mask.png"
    cv2.imwrite(output_path, mask_vis)
    print(f"Diagnostic mask saved to {output_path}")
    print(f"WDS Filter elements: {len(boxes)}")

if __name__ == "__main__":
    show_mask()
