import os
import sys
import torch
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(ROOT, "weights/icon_detect/model.pt")
CAPTION_PATH = os.path.join(ROOT, "weights/icon_caption_florence")

def test_load():
    try:
        print(f"Testing YOLO load from: {YOLO_PATH}")
        yolo = YOLO(YOLO_PATH)
        print("YOLO loaded.")
        
        print(f"Testing Florence-2 load from: {CAPTION_PATH}")
        processor = AutoProcessor.from_pretrained(CAPTION_PATH, trust_remote_code=True)
        print("Processor loaded.")
        
        model = AutoModelForCausalLM.from_pretrained(CAPTION_PATH, torch_dtype=torch.float32, trust_remote_code=True)
        print("Model loaded.")
        
        print("SUCCESS: All models loaded correctly.")
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
