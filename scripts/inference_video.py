import sys
import os
from pathlib import Path

# Add project root to path so 'models', 'configs' etc. can be found
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from models import get_model
from configs.unet_config import UNetConfig
from training.checkpoint import load_checkpoint
from data.transforms import build_transforms

def process_video(video_path, output_path, checkpoint_path):
    # 1. Setup Configuration and Model
    cfg = UNetConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    # Use the EXACT same transforms as validation
    transform = build_transforms(cfg.image_size, policy="none")
    
    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 3. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"[INFO] Processing {total_frames} frames with PADDING FIX...")
    
    # Pre-calculate padding to reverse it later
    # (Albumentations PadIfNeeded centers the image)
    scale = cfg.image_size / max(width, height)
    new_w, new_h = int(width * scale), int(height * scale)
    pad_top = (cfg.image_size - new_h) // 2
    pad_left = (cfg.image_size - new_w) // 2
    
    with torch.no_grad():
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- PRE-PROCESS (Exact Training Match) ---
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transformed = transform(image=img_rgb)
            img_tensor = transformed["image"].unsqueeze(0).to(device)
            
            # --- INFERENCE ---
            logits = model(img_tensor)
            mask = torch.sigmoid(logits) > 0.5
            mask = mask.cpu().numpy().squeeze().astype(np.uint8)
            
            # --- ALIGNMENT FIX (Remove Padding) ---
            # Extract the actual image area from the square mask
            mask_cropped = mask[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
            
            # Resize back to original video resolution
            mask_full = cv2.resize(mask_cropped, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # --- OVERLAY ---
            green_overlay = np.zeros_like(frame)
            green_overlay[:, :, 1] = 255
            lane_color = cv2.bitwise_and(green_overlay, green_overlay, mask=mask_full)
            combined = cv2.addWeighted(frame, 1.0, lane_color, 0.4, 0)
            
            out.write(combined)
            
    cap.release()
    out.release()
    print(f"[SUCCESS] Result saved with fix: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output_lane_detection.mp4")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/e9_ablation_s42_best.pth")
    
    args = parser.parse_args()
    process_video(args.input, args.output, args.ckpt)