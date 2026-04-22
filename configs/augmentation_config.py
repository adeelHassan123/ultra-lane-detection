from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class AugConfig:
    light: Dict[str, Any] = None
    heavy: Dict[str, Any] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "light",
            {
                "horizontal_flip": 0.5,
                "brightness_contrast": 0.2,
            },
        )
        object.__setattr__(
            self,
            "heavy",
            {
                "horizontal_flip": 0.5,
                "brightness_contrast": 0.4,
                "motion_blur": 0.25,
                "gauss_noise": 0.2,
            },
        )
