"""Joint geometric transforms for RGB and four lesion maps."""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop
from PIL import Image


def _maps_to_tensor(maps: List[Image.Image]) -> torch.Tensor:
    """List of PIL grayscale -> FloatTensor [C,H,W] in [0,1]."""
    t = torch.stack([TF.pil_to_tensor(m.convert("L")).float().squeeze(0) / 255.0 for m in maps], dim=0)
    return torch.clamp(t, 0.0, 1.0)


def _geo_maps(
    maps: List[Image.Image],
    *,
    crop: Optional[Tuple[int, int, int, int]],
    img_size: int,
    hflip: bool,
    angle: float,
) -> List[Image.Image]:
    out = maps
    if crop is not None:
        i, j, th, tw = crop
        out = [
            TF.resized_crop(
                m, i, j, th, tw, (img_size, img_size), interpolation=TF.InterpolationMode.BILINEAR
            )
            for m in out
        ]
    else:
        out = [
            TF.resize(m, (img_size, img_size), interpolation=TF.InterpolationMode.BILINEAR) for m in out
        ]
    if hflip:
        out = [TF.hflip(m) for m in out]
    if abs(angle) > 1e-6:
        out = [TF.rotate(m, angle, interpolation=TF.InterpolationMode.BILINEAR) for m in out]
    return out


class JointTrainTransform:
    def __init__(
        self,
        img_size: int,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        rotation_deg: float = 15.0,
        color_jitter_b: float = 0.2,
        color_jitter_c: float = 0.2,
        color_jitter_s: float = 0.1,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.img_size = img_size
        self.crop_scale = crop_scale
        self.rotation_deg = rotation_deg
        self.mean = mean
        self.std = std
        self.cj = T.ColorJitter(
            brightness=color_jitter_b,
            contrast=color_jitter_c,
            saturation=color_jitter_s,
            hue=0.0,
        )

    def __call__(
        self,
        rgb: Image.Image,
        lesion_maps: List[Image.Image],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        i, j, th, tw = RandomResizedCrop.get_params(
            rgb, scale=self.crop_scale, ratio=(3.0 / 4.0, 4.0 / 3.0)
        )
        crop = (i, j, th, tw)
        hflip = random.random() < 0.5
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)

        rgb = TF.resized_crop(
            rgb, i, j, th, tw, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.BILINEAR
        )
        lesion_maps = _geo_maps(lesion_maps, crop=crop, img_size=self.img_size, hflip=False, angle=0.0)
        # crop already applied above via resized_crop; apply flip/rot only
        if hflip:
            rgb = TF.hflip(rgb)
            lesion_maps = [TF.hflip(m) for m in lesion_maps]
        if abs(angle) > 1e-6:
            rgb = TF.rotate(rgb, angle, interpolation=TF.InterpolationMode.BILINEAR)
            lesion_maps = [TF.rotate(m, angle, interpolation=TF.InterpolationMode.BILINEAR) for m in lesion_maps]

        rgb = self.cj(rgb)
        x = TF.to_tensor(rgb)
        x = TF.normalize(x, self.mean, self.std)
        m = torch.clamp(_maps_to_tensor(lesion_maps), 0.0, 1.0)
        return x, m


class JointEvalTransform:
    def __init__(
        self,
        img_size: int,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.img_size = img_size
        self.mean = mean
        self.std = std

    def __call__(
        self,
        rgb: Image.Image,
        lesion_maps: List[Image.Image],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb = TF.resize(rgb, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.BILINEAR)
        lesion_maps = [
            TF.resize(m, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.BILINEAR)
            for m in lesion_maps
        ]
        x = TF.to_tensor(rgb)
        x = TF.normalize(x, self.mean, self.std)
        m = torch.clamp(_maps_to_tensor(lesion_maps), 0.0, 1.0)
        return x, m
