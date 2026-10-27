"""Multi-task loss for Clinical-LQF."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads import ordinal_targets, regression_target


def class_balanced_weights(
    labels: torch.Tensor,
    num_classes: int,
    mode: str = "sqrt",
    effective_beta: float = 0.9999,
) -> torch.Tensor:
    """Per-class CE weights normalized to mean 1.

    mode:
      none/uniform — all ones (use when relying on WeightedRandomSampler)
      sqrt         — 1/sqrt(freq) (smoothed; common long-tail heuristic)
      inverse      — 1/freq (strong; avoid combining with an oversampling sampler)
      effective    — (1-beta)/(1-beta^n), Cui et al. class-balanced loss
    """
    device = labels.device
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    m = str(mode).lower().strip()
    if m in ("none", "uniform", "off", "1"):
        w = torch.ones(num_classes, device=device, dtype=torch.float32)
        return w
    if m == "inverse":
        w = 1.0 / counts
    elif m in ("effective", "cb", "cbce"):
        beta = float(effective_beta)
        w = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    else:
        w = 1.0 / torch.sqrt(counts)
    w = w / w.mean()
    return w.to(device)


def per_sample_weights(labels: torch.Tensor, num_classes: int, mode: str = "inverse") -> torch.Tensor:
    """Sample weights for WeightedRandomSampler (one weight per training row)."""
    cw = class_balanced_weights(labels, num_classes, mode=mode)
    return cw[labels.long()]


def relation_soft_labels(labels: torch.Tensor, tau: float, num_classes: int) -> torch.Tensor:
    j = torch.arange(num_classes, device=labels.device, dtype=torch.float32)
    diff = torch.abs(j.view(1, -1) - labels.float().view(-1, 1))
    q = torch.exp(-diff / tau)
    return q / q.sum(dim=1, keepdim=True)


class ClinicalLQFTotalLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        w_cls: float = 1.0,
        w_ord: float = 0.4,
        w_reg: float = 0.4,
        w_relation: float = 0.2,
        w_burden: float = 0.1,
        relation_tau: float = 0.7,
        ce_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.w_cls = w_cls
        self.w_ord = w_ord
        self.w_reg = w_reg
        self.w_relation = w_relation
        self.w_burden = w_burden
        self.relation_tau = relation_tau
        self.register_buffer("ce_weight", ce_class_weights if ce_class_weights is not None else torch.ones(num_classes))

        self.ce = nn.CrossEntropyLoss(weight=self.ce_weight)
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        logits_cls = out["logits_cls"]
        logits_ord = out["logits_ord"]
        score_reg = out["score_reg"]
        burden_pred = out["burden_pred"]
        labels = batch["label"].to(logits_cls.device)
        burden_tgt = batch["burden_target"].to(logits_cls.device, dtype=torch.float32)

        L_cls = self.ce(logits_cls, labels)
        ord_t = ordinal_targets(labels)
        L_ord = self.bce(logits_ord, ord_t)
        L_reg = self.smooth_l1(score_reg, regression_target(labels))
        q = relation_soft_labels(labels, self.relation_tau, self.num_classes)
        log_p = F.log_softmax(logits_cls, dim=1)
        L_rel = self.kl(log_p, q)
        L_burden = self.smooth_l1(burden_pred, burden_tgt)

        total = (
            self.w_cls * L_cls
            + self.w_ord * L_ord
            + self.w_reg * L_reg
            + self.w_relation * L_rel
            + self.w_burden * L_burden
        )
        parts = {
            "L_cls": L_cls.detach(),
            "L_ord": L_ord.detach(),
            "L_reg": L_reg.detach(),
            "L_rel": L_rel.detach(),
            "L_burden": L_burden.detach(),
        }
        return total, parts
