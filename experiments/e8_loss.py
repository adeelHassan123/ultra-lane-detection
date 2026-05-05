"""
E8: Loss Function Comparison — DiceBCE vs Combined (Tversky + BCE)
===================================================================
Variable changed:  loss_name → "dice_bce"  (vs "combined" in E7)
Hypothesis: DiceBCE weights false negatives and false positives equally (both
            in α=β=0.5), whereas the combined Tversky loss penalises missed
            lanes (FN) 2.3× more than false alarms (FP). On a sparse lane
            dataset this asymmetry should make Tversky clearly superior.
Available loss names: "bce", "dice", "tversky", "combined"
Expected outcome: dice_bce < combined, validating the Tversky-weighted choice.
Architecture: U-Net with ResNet-34 encoder (ImageNet pretrained)
              Same architecture as E7, only loss function changes.
Baseline (E0) mean IoU: 0.4968 | E7 target: ≥0.60
"""

from dataclasses import replace

from configs.unet_config import UNetConfig


def get_config():
    base = UNetConfig(experiment_name="e8_loss")
    return replace(
        base,
        pretrained=True,
        train_augmentation="heavy",
        loss_name="dice_bce",               # Changed: "combined" → "dice_bce"
        freeze_encoder_epochs=15,
        encoder_learning_rate=1e-4,
        decoder_learning_rate=1e-3,
    )
