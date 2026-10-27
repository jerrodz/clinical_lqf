"""Export timm ImageNet encoder to local .pth for offline DDR training.

Saves **features_only** backbone weights (matches ClinicalLQFNet / ConvNeXtEncoder at train time).

Uses the default Hugging Face / timm download path. Set HF_ENDPOINT yourself if you need a mirror
(e.g. export HF_ENDPOINT=https://hf-mirror.com).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "pretrained" / "convnext_base.fb_in22k_ft_in1k_384_encoder.pth"
DEFAULT_OUT_INDICES = (1, 2, 3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-name",
        default="convnext_base.fb_in22k_ft_in1k_384",
        help="timm model name (must match training config backbone_name)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--out-indices",
        default="1,2,3",
        help="Comma-separated feature stages (must match config model.out_indices)",
    )
    args = p.parse_args()
    out_indices = tuple(int(x.strip()) for x in args.out_indices.split(","))

    ep = os.environ.get("HF_ENDPOINT", "").strip()
    if ep:
        print(f"HF_ENDPOINT = {ep}")
    else:
        print("HF_ENDPOINT unset (using default Hugging Face / timm download)")

    print(
        f"Downloading/loading timm {args.model_name} "
        f"(pretrained=True, features_only=True, out_indices={out_indices})..."
    )
    try:
        m = timm.create_model(
            args.model_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
        )
    except Exception as e:
        raise SystemExit(
            f"Failed to load {args.model_name}: {e}\n"
            "Optional mirror (user-set): HF_ENDPOINT=https://hf-mirror.com "
            f"python tools/export_convnext_pretrained.py --model-name {args.model_name} --out {args.out}"
        ) from e
    sd = m.state_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.out)
    print(f"Saved {len(sd)} tensors -> {args.out}")


if __name__ == "__main__":
    main()
