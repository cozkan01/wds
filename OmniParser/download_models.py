import os
import torch
from huggingface_hub import snapshot_download
from ultralytics import YOLO

def download_models():
    weights_parent = "weights"
    icon_detect_dir = os.path.join(weights_parent, "icon_detect")
    icon_caption_dir = os.path.join(weights_parent, "icon_caption_florence")
    
    os.makedirs(icon_detect_dir, exist_ok=True)
    os.makedirs(icon_caption_dir, exist_ok=True)

    yolo_hf_repo = "microsoft/OmniParser-v2.0" 
    yolo_local_path = os.path.join(icon_detect_dir, "model.pt")
    
    if not os.path.exists(yolo_local_path):
        print(f"Downloading YOLO model to {yolo_local_path}...")
        try:
            snapshot_download(
                repo_id=yolo_hf_repo,
                allow_patterns=["icon_detect/*"],
                local_dir=weights_parent
            )
            print("YOLO model downloaded.")
        except Exception as e:
            print(f"Failed to download")

    florence_repo = "microsoft/Florence-2-base"
    if not os.listdir(icon_caption_dir):
        print(f"Downloading Florence-2 model to {icon_caption_dir}...")
        try:
            snapshot_download(
                repo_id=florence_repo,
                local_dir=icon_caption_dir
            )
            print("Florence-2 model downloaded.")
        except Exception as e:
            print(f"Failed to download")

    print(f"Complete")

if __name__ == "__main__":
    download_models()
