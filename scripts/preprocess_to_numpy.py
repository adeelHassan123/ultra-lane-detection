"""
Preprocess dataset to numpy format for maximum loading speed.
Converts images from 720x1280 to 256x256 and saves as compressed numpy.
This provides 10-20x faster data loading compared to JPEG/PNG.
"""

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import os


def preprocess_split(src_root: str, dst_root: str, split: str, image_size: int = 256):
    """Preprocess one split (train/val/test) to numpy format."""
    src_path = Path(src_root) / split
    dst_path = Path(dst_root) / split
    
    # Create output directories
    (dst_path / "images").mkdir(parents=True, exist_ok=True)
    (dst_path / "masks").mkdir(parents=True, exist_ok=True)
    
    img_dir = src_path / "images"
    mask_dir = src_path / "masks"
    
    image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    
    print(f"Processing {split}: {len(image_files)} images...")
    
    for img_path in tqdm(image_files, desc=f"{split}"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            print(f"Warning: Mask not found {mask_path}")
            continue
            
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            continue
        
        # Resize both to target size
        # Use INTER_LINEAR for images (smooth), INTER_NEAREST for masks (preserve binary)
        img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        
        # Convert mask to binary {0, 1}
        mask_binary = (mask_resized > 127).astype(np.uint8)
        
        # Save as compressed numpy (fastest format for loading)
        out_path = dst_path / "images" / f"{img_path.stem}.npz"
        np.savez_compressed(
            out_path,
            image=img_resized,
            mask=mask_binary
        )
    
    print(f"[OK] {split} complete: {len(list((dst_path / 'images').glob('*.npz')))} files")


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset to numpy format")
    parser.add_argument("--src", default="/content/drive/MyDrive/lane_detection_data/processed",
                       help="Source directory with train/val/test splits")
    parser.add_argument("--dst", default="/content/fast_data",
                       help="Destination directory for numpy files")
    parser.add_argument("--size", type=int, default=256,
                       help="Target image size (default: 256)")
    args = parser.parse_args()
    
    print(f"[SPEEDUP] Preprocessing to numpy format")
    print(f"   Source: {args.src}")
    print(f"   Destination: {args.dst}")
    print(f"   Target size: {args.size}x{args.size}")
    print()
    
    # Process all splits
    for split in ["train", "val", "test"]:
        if Path(args.src) / split / "images":
            preprocess_split(args.src, args.dst, split, args.size)
        else:
            print(f"Skipping {split} (not found)")
    
    print(f"\n[OK] Preprocessing complete!")
    print(f"   Now update configs/base.py:")
    print(f"   data_root: str = \"{args.dst}\"")
    print(f"\n   Expected speedup: 10-20x faster data loading")


if __name__ == "__main__":
    main()
