import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _base_ops(size: int):
    return [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]


def build_transforms(image_size=256, policy: str = "none"):
    ops = []

    if policy == "light":
        ops.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )
    elif policy == "heavy":
        ops.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.MotionBlur(p=0.25),
                A.GaussNoise(p=0.2),
            ]
        )

    return A.Compose(ops + _base_ops(image_size))
