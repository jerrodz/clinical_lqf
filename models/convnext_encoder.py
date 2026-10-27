"""ConvNeXt multi-scale feature extractor via timm."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import timm


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        out_indices: Sequence[int],
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.out_indices = tuple(out_indices)
        if len(self.out_indices) != 3:
            raise ValueError("Clinical-LQF requires three encoder feature levels")
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True,
            out_indices=self.out_indices, in_chans=3,
        )
        ch = self.backbone.feature_info.channels()
        self._channels: List[int] = list(ch)

    @property
    def level_channels(self) -> List[int]:
        return self._channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)
