import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataloader import build_dataloaders
from losses import get_loss
from metrics.advanced import auc_iou
from metrics.segmentation import dice_score
from models import get_model
from training.checkpoint import load_checkpoint, save_checkpoint
from training.early_stopping import EarlyStopping
from training.train_epoch import train_one_epoch
from training.validate_epoch import validate
from utils.device import get_device
from utils.logger import CSVLogger
from utils.seed import set_all_seeds


class Trainer:
    def __init__(self, cfg):
        set_all_seeds(cfg.seed)
        self.cfg = cfg
        self.cfg.ensure_output_dirs()
        self.device = get_device()
        model = get_model(cfg).to(self.device)
        # Compile model for 20-30% speedup (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("[OPTIMIZATION] Compiling model with torch.compile() for maximum speed...")
            model = torch.compile(model, mode="max-autotune")
        self.model = model
        self.criterion = get_loss(cfg)
        self.optimizer = self._build_optimizer(phase="warmup")
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and self.device.type == "cuda"))
        self.stopper = EarlyStopping(patience=cfg.patience)
        self.train_loader, self.val_loader = build_dataloaders(cfg)
        self.logger = CSVLogger(f"{cfg.logs_dir}/{cfg.experiment_name}_s{cfg.seed}.csv")
        self.checkpoint_path = f"{self.cfg.checkpoints_dir}/{self.cfg.experiment_name}_s{self.cfg.seed}_best.pth"

    def _build_optimizer(self, phase: str = "full"):
        if (
            self.cfg.model_name == "unet_smp"
            and self.cfg.pretrained
            and hasattr(self.model, "freeze_encoder")
            and phase == "warmup"
        ):
            self.model.freeze_encoder()
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            return AdamW(params, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

        if (
            self.cfg.model_name == "unet_smp"
            and self.cfg.pretrained
            and hasattr(self.model, "get_differential_param_groups")
        ):
            self.model.unfreeze_encoder()
            param_groups = self.model.get_differential_param_groups(
                self.cfg.encoder_learning_rate,
                self.cfg.decoder_learning_rate,
            )
            return AdamW(param_groups, weight_decay=self.cfg.weight_decay)

        if self.cfg.optimizer_name == "sgd":
            return SGD(self.model.parameters(), lr=self.cfg.learning_rate, momentum=0.9, weight_decay=self.cfg.weight_decay)
        if self.cfg.optimizer_name == "adam":
            return Adam(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        return AdamW(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)

    def fit(self):
        start_epoch, best_iou = load_checkpoint(
            self.checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
        )
        for epoch in range(start_epoch, self.cfg.epochs):
            if (
                self.cfg.model_name == "unet_smp"
                and self.cfg.pretrained
                and epoch == self.cfg.freeze_encoder_epochs
            ):
                self.optimizer = self._build_optimizer(phase="full")
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(self.cfg.epochs - epoch, 1))

            train_loss, train_iou = train_one_epoch(
                self.model, self.train_loader, self.criterion, self.optimizer, self.scaler, self.device, self.cfg
            )
            val_loss, val_iou = validate(self.model, self.val_loader, self.criterion, self.device, self.cfg)
            val_auc, val_dice = self._compute_extra_validation_metrics()
            self.scheduler.step()
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_auc_iou": val_auc,
                "val_dice": val_dice,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.logger.log(row)
            if val_iou > best_iou:
                best_iou = val_iou
                save_checkpoint(
                    self.checkpoint_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    best_iou,
                )
            if (epoch + 1) % self.cfg.save_every == 0:
                save_checkpoint(
                    f"{self.cfg.checkpoints_dir}/{self.cfg.experiment_name}_s{self.cfg.seed}_ep{epoch+1}.pth",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    best_iou,
                )
            if self.stopper.step(val_iou):
                break
        return best_iou

    @torch.no_grad()
    def _compute_extra_validation_metrics(self):
        self.model.eval()
        auc_values = []
        dice_values = []
        for images, masks in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            logits = self.model(images)
            auc_values.append(auc_iou(logits, masks))
            dice_values.append(dice_score(logits, masks, threshold=self.cfg.threshold))
        if not auc_values:
            return 0.0, 0.0
        return sum(auc_values) / len(auc_values), sum(dice_values) / len(dice_values)
