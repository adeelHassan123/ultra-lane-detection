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

def process_video(video_path, output_path, checkpoint_path, threshold=0.4):
    # 1. Setup Architecture
    exp_id = Path(checkpoint_path).name.split('_')[0]
    cfg = CNNConfig() if exp_id in ["e0", "e1", "e2", "e3", "e4", "e5"] else UNetConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
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
    
    # Pre-calculate padding
    scale = cfg.image_size / max(width, height)
    new_w, new_h = int(width * scale), int(height * scale)
    pad_top = (cfg.image_size - new_h) // 2
    pad_left = (cfg.image_size - new_w) // 2
    
    # --- TEMPORAL SMOOTHING BUFFER ---
    prev_mask = None
    
    print(f"[INFO] Generating PORTFOLIO-GRADE video using {exp_id}...")

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
            
            # --- ROI MASKING (Industry Secret #1) ---
            # Most dashcams have lanes in the bottom 60%. Ignore the top 40% (sky).
            roi_limit = int(new_h * 0.45)
            probs[:pad_top + roi_limit, :] = 0
            
            # --- THRESHOLD ---
            mask = (probs > threshold).astype(np.uint8)
            
            # --- LANE THINNING (Industry Secret #2) ---
            # Shrink the 'fat' dilated training masks back to real line sizes
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            # Remove isolated noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            
            # --- TEMPORAL SMOOTHING (Industry Secret #3) ---
            # Average with previous frame to stop flickering
            if prev_mask is not None:
                mask = cv2.bitwise_or(mask, cv2.bitwise_and(mask, prev_mask))
            prev_mask = mask.copy()
            
            # --- ALIGNMENT & RESIZE ---
            mask_cropped = mask[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
            mask_full = cv2.resize(mask_cropped, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # --- BEAUTIFUL VISUALIZATION ---
            # We will use a neon-green "Glow" look
            canvas = frame.copy()
            
            # Create a thick "glow" layer
            glow_mask = cv2.dilate(mask_full, np.ones((7,7), np.uint8), iterations=1)
            canvas[glow_mask > 0] = [0, 255, 0] # Bright Green
            
            # Add text overlay to make it look professional
            cv2.putText(canvas, f"AI LANE DETECTION | MODEL: {exp_id.upper()}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Blend with original (alpha=0.6 for the lane)
            combined = cv2.addWeighted(frame, 1.0, canvas, 0.4, 0)
            
            out.write(combined)
            
    cap.release()
    out.release()
    print(f"[SUCCESS] Portfolio video saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="portfolio_demo.mp4")
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/e7_transfer_s42_best.pth")
    parser.add_argument("--thresh", type=float, default=0.35) 
    
    args = parser.parse_args()
    process_video(args.input, args.output, args.ckpt, args.thresh)
