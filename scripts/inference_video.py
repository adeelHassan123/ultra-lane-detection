import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from models import get_model
from configs.unet_config import UNetConfig
from configs.cnn_config import CNNConfig
from training.checkpoint import load_checkpoint
from data.transforms import build_transforms

def process_video(video_path, output_path, checkpoint_path, threshold=0.5):
    # 1. Setup Architecture
    exp_id = Path(checkpoint_path).name.split('_')[0]
    cfg = CNNConfig() if exp_id in ["e0", "e1", "e2", "e3", "e4", "e5"] else UNetConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    # Validation transforms (no noise, just resize/pad/normalize)
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
    
    # Alignment Math
    scale = cfg.image_size / max(width, height)
    new_w, new_h = int(width * scale), int(height * scale)
    pad_top = (cfg.image_size - new_h) // 2
    pad_left = (cfg.image_size - new_w) // 2
    
    print(f"[INFO] Running ROBUST inference using {exp_id}...")

    with torch.no_grad():
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret: break
            
            # --- PRE-PROCESS ---
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transformed = transform(image=img_rgb)
            img_tensor = transformed["image"].unsqueeze(0).to(device)
            
            # --- INFERENCE ---
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            
            # --- ROBUST POST-PROCESS ---
            # 1. Threshold to remove low-confidence noise (hallucinations)
            mask = (probs > threshold).astype(np.uint8)
            
            # 2. Denoise: Remove small isolated dots
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 3. Alignment Fix: Crop the padding
            mask_cropped = mask[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
            probs_cropped = probs[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
            
            # 4. Resize back to video size
            mask_full = cv2.resize(mask_cropped, (width, height), interpolation=cv2.INTER_NEAREST)
            probs_full = cv2.resize(probs_cropped, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # --- SMOOTH VISUALIZATION ---
            # Create a heatmap look (Green color intensity based on probability)
            heatmap = (probs_full * 255).astype(np.uint8)
            green_lane = np.zeros_like(frame)
            green_lane[:, :, 1] = heatmap # Apply confidence to green channel
            
            # Highlight only the mask area
            final_lane = cv2.bitwise_and(green_lane, green_lane, mask=mask_full)
            
            # Blend: We use a lower alpha for the lane to keep it clean
            combined = cv2.addWeighted(frame, 1.0, final_lane, 0.6, 0)
            
            out.write(combined)
            
    cap.release()
    out.release()
    print(f"[SUCCESS] Robust result saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="robust_demo.mp4")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/e7_transfer_s42_best.pth")
    parser.add_argument("--thresh", type=float, default=0.4) # Lower thresh can help if model is shy
    
    args = parser.parse_args()
    process_video(args.input, args.output, args.ckpt, args.thresh)
