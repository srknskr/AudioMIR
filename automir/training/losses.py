from typing import Dict, Optional
import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining Tempo Regression (SmoothL1), Style Classification (CrossEntropy),

    and optional Meter Classification (CrossEntropy) with configurable task weights.
    """

    def __init__(
        self,
        weight_tempo: float = 1.0,
        weight_style: float = 1.0,
        weight_meter: float = 0.5,
        enable_meter: bool = True,
    ):
        super().__init__()
        self.weight_tempo = weight_tempo
        self.weight_style = weight_style
        self.weight_meter = weight_meter
        self.enable_meter = enable_meter

        self.tempo_criterion = nn.SmoothL1Loss()
        self.style_criterion = nn.CrossEntropyLoss()
        self.meter_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pred_log2_bpm = predictions["log2_bpm"]
        target_log2_bpm = targets["log2_bpm"]
        loss_tempo = self.tempo_criterion(pred_log2_bpm, target_log2_bpm)

        pred_style_logits = predictions["style_logits"]
        target_style_id = targets["style_id"]
        loss_style = self.style_criterion(pred_style_logits, target_style_id)

        total_loss = self.weight_tempo * loss_tempo + self.weight_style * loss_style

        loss_dict = {
            "total_loss": total_loss,
            "loss_tempo": loss_tempo.detach(),
            "loss_style": loss_style.detach(),
        }

        if self.enable_meter and "meter_logits" in predictions and "meter_id" in targets:
            loss_meter = self.meter_criterion(predictions["meter_logits"], targets["meter_id"])
            total_loss = total_loss + self.weight_meter * loss_meter
            loss_dict["total_loss"] = total_loss
            loss_dict["loss_meter"] = loss_meter.detach()

        return loss_dict
