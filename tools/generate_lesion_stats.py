"""Compute lesion_stats.csv from MA/HE/EX/SE probability maps (.npy)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from datasets.ddr_dataset import STAT_COLUMNS
from utils.package_paths import expand_path_string

LESION_ORDER = ["MA", "HE", "EX", "SE"]


def load_map(path: str) -> np.ndarray:
    path = expand_path_string(path)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Missing lesion map: {path!r}")
    a = np.load(path, allow_pickle=False).astype(np.float32)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError(f"Lesion map must be a finite, nonempty 2D array: {path}")
    if a.min() < 0 or a.max() > 1:
        raise ValueError(f"Lesion map probabilities must be in [0, 1]: {path}")
    return a


def component_count(p: np.ndarray, thr: float = 0.5) -> int:
    m = (p >= thr).astype(np.int32)
    if m.sum() == 0:
        return 0
    _, n = ndimage.label(m)
    return int(n)


def resize_map(a: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if a.shape == (h, w):
        return a
    img = Image.fromarray((np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def stats_for_maps(maps: dict[str, np.ndarray]) -> dict[str, float]:
    stacked = []
    out: dict[str, float] = {}
    total_area = 0.0
    for k in LESION_ORDER:
        p = maps[k]
        # Area ratio uses soft probability values.
        out[f"{k}_area_ratio"] = float(p.mean())
        out[f"{k}_mean_prob"] = float(p.mean())
        out[f"{k}_max_prob"] = float(p.max())
        total_area += float(p.mean())
        stacked.append(p)
        out[f"{k}_component_count"] = float(component_count(p))
    stacked_np = np.stack(stacked, axis=0) + 1e-8
    norm = stacked_np / stacked_np.sum(axis=0, keepdims=True)
    ent = -(norm * np.log(norm)).sum(axis=0)
    out["total_lesion_area"] = total_area
    out["lesion_entropy"] = float(ent.mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master-csv", type=str, required=True)
    p.add_argument("--out", type=str, default="lesion_stats.csv")
    args = p.parse_args()

    df = pd.read_csv(args.master_csv)
    rows = []
    for _, row in df.iterrows():
        iid = str(row["image_id"])
        maps = {}
        for k in LESION_ORDER:
            col = f"{k}_pred_map_path"
            path = str(row[col]) if col in row and pd.notna(row[col]) else ""
            maps[k] = load_map(path)
        sh = maps["MA"].shape
        for k in LESION_ORDER:
            maps[k] = resize_map(maps[k], sh)
        st = stats_for_maps(maps)
        st["image_id"] = iid
        rows.append(st)

    out_df = pd.DataFrame(rows)
    if out_df["image_id"].duplicated().any():
        raise SystemExit("duplicate image_id in generated lesion_stats")
    cols = ["image_id"] + [c for c in STAT_COLUMNS if c in out_df.columns]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df[cols].to_csv(args.out, index=False)
    print("wrote", args.out, out_df.shape)


if __name__ == "__main__":
    main()
