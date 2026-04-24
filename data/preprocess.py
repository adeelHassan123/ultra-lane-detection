from pathlib import Path
import shutil

import cv2
import numpy as np


DILATION_KERNEL_SIZE = 15


def copy_images(src_root: str, dst_root: str) -> None:
    for split in ("train", "val"):
        in_dir = Path(src_root) / split / "images"
        out_dir = Path(dst_root) / split / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(in_dir.glob("*.*")):
            shutil.copy2(img_path, out_dir / img_path.name)


def dilate_masks(src_root: str, dst_root: str) -> None:
    """
    One-time preprocessing to fix thin lane masks.
    """
    kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)

    for split in ("train", "val"):
        in_dir = Path(src_root) / split / "masks"
        out_dir = Path(dst_root) / split / "masks"
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in sorted(in_dir.glob("*.png")):
            mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            mask = (mask > 127).astype(np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            ratio = dilated.sum() / dilated.size
            if not (0.02 < ratio < 0.12):
                print(f"WARNING: unusual lane ratio {ratio:.3f} in {f}")
            cv2.imwrite(str(out_dir / f.name), dilated * 255)

    print("[preprocess] Mask dilation complete.")
