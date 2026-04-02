import os
import sys
import cv2
import time
import torch

import wds_omniparser_bridge as bridge

TEST_IMAGE = "test1.png"

def run_test():
    if not os.path.exists(TEST_IMAGE):
        print(f"Test image {TEST_IMAGE} not found. Searching for 'clipboard_temp.png'...")
        TEST_IMAGE = "clipboard_temp.png"
        if not os.path.exists(TEST_IMAGE):
            print("No test image found. Please provide test1.png or clipboard_temp.png")
            return

    print(f"--- Running WDS Grounding Inference Test on {TEST_IMAGE} ---")
    
    try:
        print("Loading YOLO...")
        yolo = bridge.get_yolo_model(bridge.YOLO_PATH)
        print("Loading Florence-2...")
        caption = bridge.get_caption_model_processor("florence2", bridge.CAPTION_PATH)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Failed to load models: {e}")
        print("Check if weights/ directory contains necessary files.")
        return

    print("\nStarting full pipeline...")
    start_t = time.time()
    try:
        bridge.main()
        end_t = time.time()
        print(f"\nPipeline finished in {end_t - start_t:.1f} seconds.")
        print(f"Output saved to {bridge.OUTPUT_PATH}")
        
        if os.path.exists(bridge.OUTPUT_PATH):
            print("Inference Test: SUCCESS")
        else:
            print("Inference Test: FAILED (Output file not found)")
            
    except Exception as e:
        print(f"Inference pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
