import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from automir.training.losses import MultiTaskLoss
from automir.evaluation.metrics import (
    compute_tempo_metrics,
    compute_classification_metrics,
)
from automir.models.heads import TempoRegressionHead


class Trainer:
    """Multi-task model trainer with validation, early stopping, and crash protection."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device: torch.device,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_weights: Optional[Dict[str, float]] = None,
        enable_meter: bool = True,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        w_dict = loss_weights or {"tempo": 1.0, "style": 1.0, "meter": 0.5}
        self.criterion = MultiTaskLoss(
            weight_tempo=w_dict.get("tempo", 1.0),
            weight_style=w_dict.get("style", 1.0),
            weight_meter=w_dict.get("meter", 0.5),
            enable_meter=enable_meter,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(self.train_dataset) > self.batch_size,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def _prepare_inputs(self, batch: Dict[str, Any]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if "logmel" in batch and "tempogram" in batch:
            return batch["logmel"].to(self.device), batch["tempogram"].to(self.device)
        return batch["feature"].to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            inputs = self._prepare_inputs(batch)
            targets = {
                "log2_bpm": batch["log2_bpm"].to(self.device),
                "style_id": batch["style_id"].to(self.device),
            }
            if "meter_id" in batch:
                targets["meter_id"] = batch["meter_id"].to(self.device)

            self.optimizer.zero_grad()

            if isinstance(inputs, tuple):
                preds = self.model(inputs[0], inputs[1])
            else:
                preds = self.model(inputs)

            loss_out = self.criterion(preds, targets)
            loss = loss_out["total_loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Training produced NaN or Inf loss.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        self.model.eval()
        loader = dataloader or self.val_loader

        pred_bpms, target_bpms = [], []
        pred_styles, target_styles = [], []
        pred_meters, target_meters = [], []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                inputs = self._prepare_inputs(batch)
                targets = {
                    "log2_bpm": batch["log2_bpm"].to(self.device),
                    "style_id": batch["style_id"].to(self.device),
                }
                if "meter_id" in batch:
                    targets["meter_id"] = batch["meter_id"].to(self.device)

                if isinstance(inputs, tuple):
                    preds = self.model(inputs[0], inputs[1])
                else:
                    preds = self.model(inputs)

                loss_out = self.criterion(preds, targets)
                total_loss += loss_out["total_loss"].item()
                n_batches += 1

                # BPM conversion
                pred_linear_bpm = TempoRegressionHead.to_linear_bpm(preds["log2_bpm"])
                pred_bpms.extend(pred_linear_bpm.cpu().numpy().tolist())
                target_bpms.extend(batch["bpm"].numpy().tolist())

                # Style predictions
                s_logits = preds["style_logits"]
                s_preds = torch.argmax(s_logits, dim=-1)
                pred_styles.extend(s_preds.cpu().numpy().tolist())
                target_styles.extend(batch["style_id"].numpy().tolist())

                # Meter predictions
                if "meter_logits" in preds and "meter_id" in batch:
                    m_logits = preds["meter_logits"]
                    m_preds = torch.argmax(m_logits, dim=-1)
                    pred_meters.extend(m_preds.cpu().numpy().tolist())
                    target_meters.extend(batch["meter_id"].numpy().tolist())

        val_loss = total_loss / max(n_batches, 1)
        tempo_metrics = compute_tempo_metrics(pred_bpms, target_bpms)
        style_metrics = compute_classification_metrics(pred_styles, target_styles, prefix="style")

        result = {
            "val_loss": round(val_loss, 4),
            **tempo_metrics,
            **style_metrics,
        }

        if pred_meters:
            meter_metrics = compute_classification_metrics(pred_meters, target_meters, prefix="meter")
            result.update(meter_metrics)

        return result

    def fit(
        self,
        epochs: int = 10,
        patience: int = 5,
    ) -> Dict[str, Any]:
        """Train model across epochs with early stopping on validation loss."""
        best_val_loss = float("inf")
        best_metrics = {}
        no_improve = 0

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(epochs, 1), eta_min=1e-5
        )

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()
            scheduler.step()

            val_loss = val_metrics["val_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = {**val_metrics, "best_epoch": epoch}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return best_metrics
