"""DDR grading: RGB, four aligned lesion probability maps and 18 statistics."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.package_paths import expand_path_string

MAP_ORDER = ("MA", "HE", "EX", "SE")

STAT_COLUMNS = [
    "MA_area_ratio",
    "HE_area_ratio",
    "EX_area_ratio",
    "SE_area_ratio",
    "MA_mean_prob",
    "HE_mean_prob",
    "EX_mean_prob",
    "SE_mean_prob",
    "MA_max_prob",
    "HE_max_prob",
    "EX_max_prob",
    "SE_max_prob",
    "total_lesion_area",
    "lesion_entropy",
    "MA_component_count",
    "HE_component_count",
    "EX_component_count",
    "SE_component_count",
]


class DDRGradingMasterDataset(Dataset):
    def __init__(self, csv_path: str, split: str, transform, stats_csv: str) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split!r}")
        df = pd.read_csv(csv_path)
        required = ["image_id", "image_path", "split", "grade_label"]
        required += [f"{k}_pred_map_path" for k in MAP_ORDER]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Master CSV missing columns: {missing}")
        if df["image_id"].isna().any() or df["image_id"].duplicated().any():
            raise ValueError("Master CSV image_id must be present and globally unique across splits")
        if df["split"].isna().any() or not df["split"].isin(["train", "val", "test"]).all():
            raise ValueError("Master CSV split must be train, val or test")
        if not df["grade_label"].isin(range(5)).all():
            raise ValueError("Master CSV grade_label must be an integer from 0 to 4")
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"Master CSV has no rows for split={split!r}")
        for col in ["image_path"] + [f"{k}_pred_map_path" for k in MAP_ORDER]:
            self.df[col] = self.df[col].map(
                lambda v: expand_path_string(str(v)) if pd.notna(v) else ""
            )
        st = pd.read_csv(stats_csv)
        missing = [c for c in ["image_id"] + STAT_COLUMNS if c not in st.columns]
        if missing:
            raise ValueError(f"Stats CSV missing columns: {missing}")
        if st["image_id"].isna().any() or st["image_id"].duplicated().any():
            raise ValueError("Stats CSV image_id must be present and unique")
        self.df = self.df.merge(st[["image_id"] + STAT_COLUMNS], on="image_id", how="left", validate="one_to_one")
        stats = self.df[STAT_COLUMNS].to_numpy(dtype=np.float32)
        if not np.isfinite(stats).all():
            raise ValueError(f"Missing or non-finite statistics in split={split!r}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _load_rgb(self, path: str) -> Image.Image:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        # Prefer tifffile for TIFF (more reliable than PIL for some LZW files).
        if str(path).lower().endswith((".tif", ".tiff")):
            import tifffile

            arr = np.ascontiguousarray(tifffile.imread(path))
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] >= 4:
                arr = arr[..., :3]
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.dtype != np.uint8:
                mx = float(arr.max()) if arr.size else 0.0
                arr = (
                    np.clip(arr, 0, 255).astype(np.uint8)
                    if mx > 1.5
                    else (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                )
            return Image.fromarray(arr[..., :3], mode="RGB")
        # Fully load before returning (safer under DataLoader workers).
        with Image.open(path) as img:
            img.load()
            return img.convert("RGB")

    def _load_map(self, path: str, image_id: str, channel: str) -> Image.Image:
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Missing lesion map: image_id={image_id!r}, channel={channel}, path={path!r}")
        arr = np.load(path, allow_pickle=False).astype(np.float32)
        if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
            raise ValueError(f"Lesion map must be a finite, nonempty 2D array: {path}")
        if arr.min() < 0 or arr.max() > 1:
            raise ValueError(f"Lesion map probabilities must be in [0, 1]: {path}")
        return Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        iid = str(row["image_id"])
        rgb = self._load_rgb(str(row["image_path"]))
        maps = [self._load_map(row[f"{k}_pred_map_path"], iid, k) for k in MAP_ORDER]
        maps = [p.resize(rgb.size, Image.BILINEAR) if p.size != rgb.size else p for p in maps]
        x_rgb, p_lesion = self.transform(rgb, maps)
        return {
            "x_rgb": x_rgb,
            "p_lesion": p_lesion,
            "s_lesion": torch.tensor(row[STAT_COLUMNS].to_numpy(dtype=np.float32)),
            "label": torch.tensor(int(row["grade_label"]), dtype=torch.long),
            "burden_target": p_lesion.reshape(4, -1).mean(dim=1),
            "image_id": iid,
        }
