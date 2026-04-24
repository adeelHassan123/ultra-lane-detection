import sys
from pathlib import Path
from typing import List

import cv2


def validate_dataset(root: str) -> None:
    errors: List[str] = []
    for split in ("train", "val"):
        image_dir = Path(root) / split / "images"
        mask_dir = Path(root) / split / "masks"
        image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

        if not image_paths:
            errors.append(f"No images found in {image_dir}")
            continue

        for image_path in image_paths:
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                errors.append(f"Missing mask: {mask_path}")
                continue

            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                errors.append(f"Corrupt file pair: {image_path}, {mask_path}")
                continue

            if image.shape[:2] != mask.shape:
                errors.append(f"Shape mismatch: {image_path}")

            values = set(mask.flatten().tolist())
            if not values.issubset({0, 255}):
                errors.append(f"Mask not binary ({values}): {mask_path}")

            if int(mask.sum()) < 50 * 255:
                errors.append(f"Mask too sparse: {mask_path}")

    if errors:
        print(f"VALIDATION FAILED with {len(errors)} errors")
        for item in errors[:10]:
            print(f"- {item}")
        raise SystemExit(1)

    print(f"[validate] Dataset checks passed for {root}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m data.validate <processed_data_root>")
    validate_dataset(sys.argv[1])
