"""Build the DDR grading CSV from official split lists and four soft lesion maps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def pred_paths(image_id: str, pred_root: str) -> dict[str, str]:
    out = {}
    for k in ("MA", "HE", "EX", "SE"):
        path = Path(pred_root) / f"{image_id}_{k}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing lesion map: {path}")
        out[f"{k}_pred_map_path"] = str(path.resolve())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grading-root", type=str, required=True)
    p.add_argument("--pred-map-root", type=str, required=True, help="folder with *_MA.npy ...")
    p.add_argument("--out", type=str, default="ddr_grading_master.csv")
    args = p.parse_args()

    rows = []
    for split_txt, split_name in [("train", "train"), ("valid", "val"), ("test", "test")]:
        lst = os.path.join(args.grading_root, split_txt + ".txt")
        with open(lst, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                fname = parts[0]
                lab = int(parts[1])
                if lab == 5:
                    continue
                if lab not in range(5):
                    raise ValueError(f"Invalid DDR grade {lab}: {line}")
                image_id, _ = os.path.splitext(fname)
                img_path = os.path.abspath(os.path.join(args.grading_root, split_txt, fname))
                if not os.path.isfile(img_path):
                    raise FileNotFoundError(img_path)
                row = {
                    "image_id": image_id,
                    "image_path": img_path,
                    "grade_label": lab,
                    "split": split_name,
                }
                row.update(pred_paths(image_id, args.pred_map_root))
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty or df["image_id"].duplicated().any():
        raise ValueError("DDR split lists must contain unique image IDs and at least one graded image")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("wrote", args.out, "rows", len(df))


if __name__ == "__main__":
    main()
