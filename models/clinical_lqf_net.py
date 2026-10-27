"""Clinical-LQF: lesion-guided progressive fusion for DR grading.

Ordinal nodes N1–N4 fuse soft lesion evidence (MA/HE/EX/SE) with 18-D stats.
Lesion channel order: [0]=MA, [1]=HE, [2]=EX, [3]=SE.
DDR configuration: N2 uses HE evidence; N3 uses EX/SE evidence.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convnext_encoder import ConvNeXtEncoder

# Indices follow datasets.ddr_dataset.STAT_COLUMNS.
_IDX_MA_STATS = [0, 4, 8, 14]
_IDX_MA_HE_STATS = [0, 1, 4, 5, 8, 9, 14, 15]
_IDX_EX_SE_STATS = [2, 3, 6, 7, 10, 11, 16, 17]
STAT_DIM_DEFAULT = 18


class LesionGuidedMultiScaleExtractor(nn.Module):
    """Soft-weighted pooling of ConvNeXt F1/F2/F3 features with lesion probability maps."""

    def __init__(
        self,
        feat_dims: Dict[str, int],
        lesion_dim: int = 256,
        out_dim: int = 256,
        detach_prob: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.detach_prob = detach_prob
        c1 = feat_dims['F1']
        c2 = feat_dims['F2']
        c3 = feat_dims['F3']
        d_ma = lesion_dim + c1 + c2
        d_he = lesion_dim + c1 + c2 + c3
        d_ex = d_se = lesion_dim + c2 + c3
        self.proj_ma = nn.Sequential(
            nn.Linear(d_ma, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_he = nn.Sequential(
            nn.Linear(d_he, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_ex = nn.Sequential(
            nn.Linear(d_ex, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_se = nn.Sequential(
            nn.Linear(d_se, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def weighted_pool(feat: torch.Tensor, prob: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
        """feat [B,C,h,w], prob [B,1,H,W] → [B,C]"""
        prob = F.interpolate(prob, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        prob = prob.clamp(0.0, 1.0)
        weight = prob / (prob.sum(dim=(2, 3), keepdim=True) + eps)
        return (feat * weight).sum(dim=(2, 3))

    def forward(
        self,
        feats: List[torch.Tensor],
        p_lesion: torch.Tensor,
        z_ma: torch.Tensor,
        z_he: torch.Tensor,
        z_ex: torch.Tensor,
        z_se: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1, f2, f3 = (feats[0], feats[1], feats[2])
        p_ma = p_lesion[:, 0:1]
        p_he = p_lesion[:, 1:2]
        p_ex = p_lesion[:, 2:3]
        p_se = p_lesion[:, 3:4]
        if self.detach_prob:
            p_ma, p_he, p_ex, p_se = (p_ma.detach(), p_he.detach(), p_ex.detach(), p_se.detach())
        ma_pool = [self.weighted_pool(f1, p_ma), self.weighted_pool(f2, p_ma)]
        he_pool = [self.weighted_pool(f1, p_he), self.weighted_pool(f2, p_he), self.weighted_pool(f3, p_he)]
        ex_pool = [self.weighted_pool(f2, p_ex), self.weighted_pool(f3, p_ex)]
        se_pool = [self.weighted_pool(f2, p_se), self.weighted_pool(f3, p_se)]
        e_ma = self.proj_ma(torch.cat([z_ma, *ma_pool], dim=1))
        e_he = self.proj_he(torch.cat([z_he, *he_pool], dim=1))
        e_ex = self.proj_ex(torch.cat([z_ex, *ex_pool], dim=1))
        e_se = self.proj_se(torch.cat([z_se, *se_pool], dim=1))
        return (e_ma, e_he, e_ex, e_se)


class EvidenceFuseBlock(nn.Module):
    """Residual gated fusion block."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.gate(x) * self.proj(x)


class ClinicalLQFNet(nn.Module):
    """Clinical-LQF network with progressive N1–N4 fusion."""

    def __init__(
        self,
        backbone_name: str,
        out_indices: tuple[int, ...],
        stats_dim: int = STAT_DIM_DEFAULT,
        evidence_dim: int = 512,
        lesion_dim: int = 256,
        num_classes: int = 5,
        hr_gate_alpha: float = 0.3,
        hr_gate_detach: bool = True,
        dropout: float = 0.1,
        lesion_ms_detach_prob: bool = True,
        lesion_ms_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.evidence_dim = evidence_dim
        self.lesion_dim = lesion_dim
        self.stats_dim = stats_dim
        self.hr_gate_alpha = hr_gate_alpha
        self.hr_gate_detach = hr_gate_detach
        if stats_dim != 18 or num_classes != 5:
            raise ValueError("This DDR model requires 18 statistics and five classes")
        self.encoder = ConvNeXtEncoder(backbone_name, out_indices, pretrained=False)
        ch = self.encoder.level_channels
        c0, c1, c2 = (ch[0], ch[1], ch[-1])
        self.shallow_proj = nn.Linear(c0, evidence_dim)
        self.middle_proj = nn.Linear(c1, evidence_dim)
        self.deep_proj = nn.Linear(c2, evidence_dim)
        self.dec_stem = nn.Sequential(
            nn.Conv2d(c2, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch_conv = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 128, 1), nn.ReLU(inplace=True)) for _ in range(4)])
        self.branch_fc = nn.ModuleList([nn.Linear(128, lesion_dim) for _ in range(4)])
        self.stats_emb_ma = nn.Sequential(nn.Linear(4, 64), nn.GELU())
        self.stats_emb_ma_he = nn.Sequential(nn.Linear(8, 64), nn.GELU())
        self.stats_emb_ex_se = nn.Sequential(nn.Linear(8, 64), nn.GELU())
        self.stats_emb_global = nn.Sequential(nn.Linear(stats_dim, 128), nn.GELU())
        d_n1 = evidence_dim + lesion_dim + 64
        d_n2 = evidence_dim + evidence_dim + lesion_dim + 64
        d_n3 = evidence_dim + evidence_dim + lesion_dim * 2 + 64
        d_n4 = evidence_dim + evidence_dim + 128
        self.fuse_n1 = EvidenceFuseBlock(d_n1, evidence_dim, dropout)
        self.fuse_n2 = EvidenceFuseBlock(d_n2, evidence_dim, dropout)
        self.fuse_n3 = EvidenceFuseBlock(d_n3, evidence_dim, dropout)
        self.fuse_n4 = EvidenceFuseBlock(d_n4, evidence_dim, dropout)
        self.head_ge1 = nn.Linear(evidence_dim, 1)
        self.head_ge2 = nn.Linear(evidence_dim, 1)
        self.head_ge3 = nn.Linear(evidence_dim, 1)
        self.head_ge4 = nn.Linear(evidence_dim, 1)
        self.fc_cls = nn.Linear(evidence_dim, num_classes)
        self.fc_reg = nn.Linear(evidence_dim, 1)
        self.fc_burden = nn.Linear(evidence_dim, 4)
        self.head_top_aux = nn.Linear(evidence_dim, num_classes)
        self.head_bottom_aux = nn.Linear(evidence_dim, num_classes)
        self.ms_extractor = LesionGuidedMultiScaleExtractor(
            {"F1": ch[0], "F2": ch[1], "F3": ch[-1]}, lesion_dim=lesion_dim,
            out_dim=lesion_dim, detach_prob=lesion_ms_detach_prob, dropout=lesion_ms_dropout,
        )

    def _gather_stat_cols(self, s: torch.Tensor, indices: List[int]) -> torch.Tensor:
        return torch.stack([s[:, i] for i in indices], dim=1)

    def _gate_map(self, p_lesion: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """Spatial gate over selected lesion channels."""
        p = p_lesion
        pl = F.interpolate(p, size=hw, mode='bilinear', align_corners=False).sum(dim=1, keepdim=True)
        pl = torch.clamp(pl, 0.0, 1.0)
        if self.hr_gate_detach:
            pl = pl.detach()
        return pl

    def _decode_branches(
        self,
        feat_deep: torch.Tensor,
        p_lesion: torch.Tensor,
    ) -> List[torch.Tensor]:
        x = F.interpolate(feat_deep, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.dec_stem(x)
        if p_lesion.abs().sum() > 1e-12:
            pl = self._gate_map(p_lesion, x.shape[2:])
            x = x * (1.0 + self.hr_gate_alpha * pl)
        out: List[torch.Tensor] = []
        for i in range(4):
            y = self.branch_conv[i](x).mean(dim=(2, 3))
            out.append(self.branch_fc[i](y))
        return out

    def forward(
        self,
        x_rgb: torch.Tensor,
        p_lesion: torch.Tensor,
        s_lesion: torch.Tensor,
        return_features: bool = False,
    ):
        feats = self.encoder(x_rgb)
        f_s, f_m, f_d = (feats[0], feats[1], feats[-1])
        feat_shallow = self.shallow_proj(f_s.mean(dim=(2, 3)))
        feat_middle = self.middle_proj(f_m.mean(dim=(2, 3)))
        feat_deep = self.deep_proj(f_d.mean(dim=(2, 3)))
        s_ma = self.stats_emb_ma(self._gather_stat_cols(s_lesion, _IDX_MA_STATS))
        s_ma_he = self.stats_emb_ma_he(self._gather_stat_cols(s_lesion, _IDX_MA_HE_STATS))
        s_ex_se = self.stats_emb_ex_se(self._gather_stat_cols(s_lesion, _IDX_EX_SE_STATS))
        s_global = self.stats_emb_global(s_lesion)
        use_ms = p_lesion.abs().sum() > 1e-12
        branches = self._decode_branches(f_d, p_lesion)
        z_ma, z_he, z_ex, z_se = (branches[0], branches[1], branches[2], branches[3])
        if use_ms:
            e_ma, e_he, e_ex, e_se = self.ms_extractor([f_s, f_m, f_d], p_lesion, z_ma, z_he, z_ex, z_se)
        else:
            e_ma, e_he, e_ex, e_se = (z_ma, z_he, z_ex, z_se)
        n1 = self.fuse_n1(torch.cat([feat_shallow, e_ma, s_ma], dim=1))
        n2_in = torch.cat([n1, feat_middle, e_he, s_ma_he], dim=1)
        n2 = self.fuse_n2(n2_in)
        parts = [n2, feat_middle, e_ex, e_se, s_ex_se]
        n3 = self.fuse_n3(torch.cat(parts, dim=1))
        n4 = self.fuse_n4(torch.cat([n3, feat_deep, s_global], dim=1))
        logits_ord = torch.cat([self.head_ge1(n1), self.head_ge2(n2), self.head_ge3(n3), self.head_ge4(n4)], dim=1)
        logits_cls = self.fc_cls(n4)
        score_reg = 4.0 * torch.sigmoid(self.fc_reg(n4)).squeeze(-1)
        burden_pred = self.fc_burden(n3)
        logits_aux1 = self.head_top_aux(n1)
        logits_aux3 = self.head_bottom_aux(n3)
        out = {
            'logits_cls': logits_cls,
            'logits_ord': logits_ord,
            'score_reg': score_reg,
            'burden_pred': burden_pred,
            'logits_aux1': logits_aux1,
            'logits_aux3': logits_aux3,
        }
        if return_features:
            out['features'] = {
                'N1': n1,
                'N2': n2,
                'N3': n3,
                'N4': n4,
                'z_MA': z_ma,
                'z_HE': z_he,
                'z_EX': z_ex,
                'z_SE': z_se,
                'e_MA': e_ma,
                'e_HE': e_he,
                'e_EX': e_ex,
                'e_SE': e_se,
            }
        return out
