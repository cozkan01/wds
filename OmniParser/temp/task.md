# WDS & OmniParser Integration
- [x] Parse original image (`image.png`, `test1.png`) using `c:\js\wds\OmniParser\WDS\wehgp.py` to extract SOT bounding boxes.
- [x] Parse original image using OmniParser V2 (YOLO + Florence-2).
- [x] Compute IoU between WDS SOTs and OmniParser detections.
- [x] Identify WDS SOTs that OmniParser missed (IoU < threshold, e.g., 0.1).
- [x] Crop missing SOTs from the original image.
- [x] Run Florence-2 captioning on each cropped SOT to extract semantic meaning.
- [x] Merge the new SOTs with OmniParser's original detections.
- [x] Reconstruct and save the final annotated image (`output_combined.png`, `output_test1.png`).
- [x] Implement Click-to-Enlarge feature (Full-Size Image Viewer) in the GUI.
- [x] Refactor `gui.py` to use a `ttk.Notebook` (tabbed layout).
- [x] Create `mask_tab.py` for dedicated Laplacian mask visualization.
- [x] Connect `MaskTab` to the main analysis pipeline.
- [x] Draw SOT bounding boxes on the Laplacian Mask tab.
- [x] Eliminate gradient background (glassmorphism) false positives from WDS SOT generation.

