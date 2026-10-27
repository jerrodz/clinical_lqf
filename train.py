"""
Train and evaluate Clinical-LQF on DDR.

  python train.py --config configs/ddr_clinical_lqf.yaml

Eval only:
  python train.py --config configs/ddr_clinical_lqf.yaml --eval-only
  python train.py --config configs/ddr_clinical_lqf.yaml --eval-only --checkpoint outputs/ddr_clinical_lqf/best_model.pth

Requires: DDR master CSV, soft lesion maps (MA/HE/EX/SE), lesion_stats CSV, ConvNeXt encoder weights.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.ddr_dataset import DDRGradingMasterDataset  # noqa: E402
from datasets.transforms_joint import JointEvalTransform, JointTrainTransform  # noqa: E402
from losses.total_loss import ClinicalLQFTotalLoss, class_balanced_weights, per_sample_weights  # noqa: E402
from models.clinical_lqf_net import ClinicalLQFNet  # noqa: E402
from utils.metrics import (  # noqa: E402
    discretize_scores,
    evaluate_preds,
    score_continuous,
    search_thresholds_detail,
)


def get_encoder_backbone(module: nn.Module) -> nn.Module:
    core = module.module if isinstance(module, nn.DataParallel) else module
    return core.encoder.backbone


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths") or {}
    for key in list(paths.keys()):
        if isinstance(paths[key], str):
            paths[key] = _expand_env(paths[key])
    cfg["paths"] = paths
    return cfg


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m


class TopKCheckpointManager:
    """Keep the best one or two checkpoints by validation argmax QWK."""

    CANONICAL = ("best_model.pth", "best_model_rank2.pth")

    def __init__(
        self,
        out_dir: Path,
        k: int = 1,
    ) -> None:
        self.out_dir = out_dir
        self.k = int(k)
        if not 1 <= self.k <= len(self.CANONICAL):
            raise ValueError("keep_top_k_checkpoints must be 1 or 2")
        self.records: list[dict] = []

    def maybe_save(
        self,
        qwk: float,
        epoch: int,
        model: nn.Module,
        cfg: dict,
    ) -> bool:
        if not math.isfinite(qwk):
            return False
        if len(self.records) >= self.k and qwk <= self.records[-1]["qwk"]:
            return False

        staging = self.out_dir / f"_ckpt_ep{epoch:03d}.pth"
        payload = {
            "model": unwrap_model(model).state_dict(),
            "epoch": epoch,
            "val_qwk_argmax": qwk,
            "cfg": cfg,
        }
        torch.save(payload, staging)
        rec = {"qwk": qwk, "epoch": epoch, "path": staging}
        self.records.append(rec)
        self.records.sort(key=lambda r: r["qwk"], reverse=True)

        while len(self.records) > self.k:
            dropped = self.records.pop()
            if dropped["path"].is_file():
                dropped["path"].unlink()

        self._sync_canonical()
        return True

    def _sync_canonical(self) -> None:
        for i, name in enumerate(self.CANONICAL):
            dst = self.out_dir / name
            if i < len(self.records):
                shutil.copy2(self.records[i]["path"], dst)
            elif dst.is_file():
                dst.unlink()

        manifest = [
            {
                "rank": i + 1,
                "canonical": self.CANONICAL[i] if i < len(self.CANONICAL) else None,
                "epoch": r["epoch"],
                "val_qwk_argmax": r["qwk"],
                "staging_path": str(r["path"]),
            }
            for i, r in enumerate(self.records)
        ]
        (self.out_dir / "checkpoints_topk.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _resolve_checkpoint_path(cfg: dict, cli_ckpt: str, out_dir: Path, *, eval_only: bool) -> Path | None:
    """CLI --checkpoint > paths.init_checkpoint > (eval-only 时) output_dir/best_model.pth."""
    if cli_ckpt.strip():
        raw = _expand_env(cli_ckpt.strip())
    else:
        raw = _expand_env(str(
            cfg.get("paths", {}).get("init_checkpoint", "")
            or cfg.get("model", {}).get("init_checkpoint", "")
            or ""
        ).strip())
        if not raw and eval_only:
            cand = out_dir / "best_model.pth"
            return cand if cand.is_file() else None
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p
    for base in (ROOT, Path.cwd(), ROOT.parent):
        q = base / raw
        if q.is_file():
            return q.resolve()
    return p


def _expand_env(raw: str) -> str:
    from utils.package_paths import expand_path_string

    return expand_path_string(raw)


def _resolve_path(raw: str, out_dir: Path) -> Path | None:
    raw = _expand_env(raw)
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    for base in (ROOT, Path.cwd(), ROOT.parent, out_dir):
        q = base / raw
        if q.is_file():
            return q.resolve()
    return None


def load_pretrained_encoder_backbone(model: nn.Module, path: Path, device: torch.device) -> None:
    """Load the state dict produced by tools/export_convnext_pretrained.py."""
    state = torch.load(path, map_location=device, weights_only=True)
    unwrap_model(model).encoder.backbone.load_state_dict(state, strict=True)
    print(f"Loaded local pretrained encoder: {path}", flush=True)


def load_model_checkpoint(model: nn.Module, ckpt: dict, strict: bool | None = None) -> None:
    """Load a local Clinical-LQF model checkpoint."""
    if strict is None:
        strict = True
    incompatible = unwrap_model(model).load_state_dict(ckpt["model"], strict=strict)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"load_state_dict strict={strict} "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
        if incompatible.missing_keys:
            print("  missing (first 8):", incompatible.missing_keys[:8], flush=True)
        if incompatible.unexpected_keys:
            print("  unexpected (first 8):", incompatible.unexpected_keys[:8], flush=True)


def lr_factor(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return float(epoch + 1) / float(max(1, warmup))
    t = (epoch - warmup) / float(max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * t))


@torch.no_grad()
def run_epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    w_prob: float,
    w_reg: float,
):
    model.eval()
    all_logits = []
    all_reg = []
    all_labels = []
    all_ids = []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["x_rgb"].to(device, non_blocking=True)
        p = batch["p_lesion"].to(device, non_blocking=True)
        s = batch["s_lesion"].to(device, non_blocking=True)
        y = batch["label"].cpu().numpy()
        ids = batch["image_id"]
        with autocast(enabled=use_amp):
            out = model(x, p, s)
        logits = out["logits_cls"].float().cpu().numpy()
        sr = out["score_reg"].float().cpu().numpy()
        all_logits.append(logits)
        all_reg.append(sr)
        all_labels.append(y)
        all_ids.extend(list(ids))
    logits = np.concatenate(all_logits, axis=0)
    score_reg = np.concatenate(all_reg, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    scores = score_continuous(logits, score_reg, w_prob=w_prob, w_reg=w_reg)
    pred_argmax = logits.argmax(axis=1)
    qwk = cohen_kappa_score(labels, pred_argmax, weights="quadratic")
    return logits, score_reg, scores, labels, all_ids, float(qwk)


@torch.no_grad()
def finalize_eval_export(
    model: nn.Module,
    dl_val: DataLoader,
    dl_test: DataLoader,
    out_dir: Path,
    cfg: dict,
    device: torch.device,
    use_amp: bool,
    w_prob: float,
    w_reg_inf: float,
) -> None:
    """Fit validation thresholds and export validation/test predictions and metrics."""
    unwrap_model(model).eval()
    val_logits, val_reg, val_scores, val_labels, val_ids, _ = run_epoch_eval(
        model, dl_val, device, use_amp, w_prob, w_reg_inf
    )
    test_logits, test_reg, test_scores, test_labels, test_ids, _ = run_epoch_eval(
        model, dl_test, device, use_amp, w_prob, w_reg_inf
    )

    np.save(out_dir / "valid_logits.npy", val_logits)
    np.save(out_dir / "valid_scores.npy", val_scores)
    np.save(out_dir / "valid_labels.npy", val_labels)
    np.save(out_dir / "valid_score_reg.npy", val_reg)
    np.save(out_dir / "test_logits.npy", test_logits)
    np.save(out_dir / "test_scores.npy", test_scores)
    np.save(out_dir / "test_labels.npy", test_labels)
    np.save(out_dir / "test_score_reg.npy", test_reg)
    ts_cfg = cfg.get("threshold_search", {})
    ts_scores, ts_labels = val_scores, val_labels
    pool_desc = "val"

    n_rand = int(ts_cfg.get("n_random_samples", 4000))
    seed_ts = int(ts_cfg.get("seed", 42))
    min_pred_frac = ts_cfg.get("min_pred_frac", None)
    min_gap = float(ts_cfg.get("min_gap", 0.2))
    ts_objective = str(ts_cfg.get("objective", "mean")).lower().strip()
    ts_qwk_w = float(ts_cfg.get("qwk_weight", 0.5))
    ts_acc_w = float(ts_cfg.get("acc_weight", 0.5))
    best_t, ts_detail = search_thresholds_detail(
        ts_scores,
        ts_labels,
        n_samples=n_rand,
        seed=seed_ts,
        min_pred_frac=min_pred_frac,
        min_gap=min_gap,
        objective=ts_objective,
        qwk_weight=ts_qwk_w,
        acc_weight=ts_acc_w,
    )
    qwk_ts = float(ts_detail["qwk"])
    acc_ts = float(ts_detail["acc"])

    val_pred_t = discretize_scores(val_scores, best_t)
    test_pred_t = discretize_scores(test_scores, best_t)
    test_pred_arg = test_logits.argmax(axis=1)

    metrics = {
        "validation_argmax": evaluate_preds(val_labels, val_logits.argmax(axis=1)),
        "validation_thresholded": evaluate_preds(val_labels, val_pred_t),
        "test_argmax": evaluate_preds(test_labels, test_pred_arg),
        "test_thresholded": evaluate_preds(test_labels, test_pred_t),
        "best_thresholds": best_t.tolist(),
        "threshold_search_pool": pool_desc,
        "threshold_search_val_qwk": qwk_ts,
        "threshold_search_val_acc": acc_ts,
        "threshold_search_objective": ts_detail.get("objective", ts_objective),
        "threshold_search_score": float(ts_detail.get("score", 0.0)),
        "threshold_search_qwk_weight": ts_qwk_w,
        "threshold_search_acc_weight": ts_acc_w,
        "threshold_search_min_pred_frac": list(min_pred_frac) if min_pred_frac is not None else None,
    }
    with open(out_dir / "best_thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"t1": best_t[0], "t2": best_t[1], "t3": best_t[2], "t4": best_t[3]}, f, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({"image_id": val_ids, "label": val_labels, "pred": val_pred_t}).to_csv(
        out_dir / "valid_predictions.csv", index=False
    )
    pd.DataFrame({"image_id": test_ids, "label": test_labels, "pred": test_pred_t}).to_csv(
        out_dir / "test_predictions.csv", index=False
    )

    import matplotlib.pyplot as plt

    def plot_cm(cm, name):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(name)
        plt.colorbar(im)
        plt.savefig(out_dir / name)
        plt.close()

    plot_cm(np.array(metrics["validation_thresholded"]["confusion_matrix"]), "confusion_matrix_valid.png")
    plot_cm(np.array(metrics["test_thresholded"]["confusion_matrix"]), "confusion_matrix_test.png")

    print("eval export done. output:", out_dir, flush=True)
    print(
        f"threshold_search pool={pool_desc} objective={ts_objective} "
        f"qwk={qwk_ts:.4f} acc={acc_ts:.4f} score={float(ts_detail.get('score', 0.0)):.4f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs/ddr_clinical_lqf.yaml"))
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="Load a checkpoint and export validation/test predictions and metrics",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Local checkpoint for initialization or evaluation; evaluation defaults to output_dir/best_model.pth",
    )
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        shutil.copy(args.config, out_dir / "config.yaml")

    master_csv = Path(cfg["paths"]["master_csv"])
    stats_path = Path(cfg["paths"]["stats_csv"])
    if not master_csv.is_file():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Stats CSV not found: {stats_path}")

    img_size = int(cfg["data"]["img_size"])
    aug = cfg["augmentation"]
    train_tf = JointTrainTransform(
        img_size,
        crop_scale=tuple(aug["crop_scale"]),
        rotation_deg=float(aug["rotation_deg"]),
        color_jitter_b=float(aug["color_jitter"]["brightness"]),
        color_jitter_c=float(aug["color_jitter"]["contrast"]),
        color_jitter_s=float(aug["color_jitter"]["saturation"]),
        mean=tuple(aug["normalize_mean"]),
        std=tuple(aug["normalize_std"]),
    )
    eval_tf = JointEvalTransform(img_size, mean=tuple(aug["normalize_mean"]), std=tuple(aug["normalize_std"]))

    ce_mode = str(cfg.get("loss", {}).get("ce_weight_mode", "none")).lower()
    use_ws = bool(cfg.get("train", {}).get("use_weighted_sampler", False))
    if use_ws and ce_mode not in ("none", "uniform", "off", "1"):
        print(
            f"Warning: use_weighted_sampler=true with ce_weight_mode={ce_mode!r} double-counts "
            "class imbalance. Prefer ce_weight_mode=none when the sampler is on.",
            flush=True,
        )

    ds_kwargs = dict(stats_csv=str(stats_path))
    ds_val = DDRGradingMasterDataset(str(master_csv), "val", eval_tf, **ds_kwargs)
    ds_test = DDRGradingMasterDataset(str(master_csv), "test", eval_tf, **ds_kwargs)

    if not args.eval_only:
        ds_train = DDRGradingMasterDataset(str(master_csv), "train", train_tf, **ds_kwargs)
        tlabs = torch.tensor(ds_train.df["grade_label"].values.astype(np.int64))
        ce_mode = str(cfg.get("loss", {}).get("ce_weight_mode", "none"))
        cw = class_balanced_weights(tlabs, int(cfg["data"]["num_classes"]), mode=ce_mode)
        print(f"CE class weights ({ce_mode}): {cw.tolist()}", flush=True)
    else:
        ds_train = None  # type: ignore[assignment]
        tlabs = None  # type: ignore[assignment]
        cw = torch.ones(int(cfg["data"]["num_classes"]), device=torch.device("cpu"))

    finetune_ckpt = _resolve_checkpoint_path(cfg, args.checkpoint, out_dir, eval_only=args.eval_only)
    pretrained_enc_path = None
    if finetune_ckpt is None:
        raw = str(cfg["paths"].get("pretrained_encoder", ""))
        pretrained_enc_path = _resolve_path(raw, out_dir)
        if pretrained_enc_path is None:
            raise FileNotFoundError(f"Local pretrained encoder not found: {raw!r}")

    model = ClinicalLQFNet(
        backbone_name=cfg["model"]["backbone_name"],
        out_indices=tuple(cfg["model"]["out_indices"]),
        stats_dim=int(cfg["model"].get("stats_dim", 18)),
        evidence_dim=int(cfg["model"].get("evidence_dim", 512)),
        lesion_dim=int(cfg["model"].get("lesion_dim", 256)),
        num_classes=int(cfg["data"]["num_classes"]),
        hr_gate_alpha=float(cfg["model"].get("hr_gate_alpha", 0.3)),
        hr_gate_detach=bool(cfg["model"].get("hr_gate_detach", True)),
        dropout=float(cfg["model"].get("clinical_dropout", 0.1)),
        lesion_ms_detach_prob=bool(cfg["model"].get("lesion_ms_detach_prob", True)),
        lesion_ms_dropout=float(cfg["model"].get("lesion_ms_dropout", 0.1)),
    ).to(device)

    if pretrained_enc_path is not None:
        load_pretrained_encoder_backbone(model, pretrained_enc_path, device)

    use_dp = bool(cfg["train"].get("use_data_parallel", False)) and torch.cuda.device_count() > 1
    if use_dp:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    w_prob = float(cfg["inference"]["w_prob"])
    w_reg_inf = float(cfg["inference"]["w_reg"])

    ckpt_path = finetune_ckpt
    if ckpt_path is not None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        load_strict = True
        load_model_checkpoint(model, ckpt, strict=load_strict)
        print(
            f"Loaded checkpoint for {'eval' if args.eval_only else 'finetune'}: "
            f"{ckpt_path} (epoch {ckpt.get('epoch', '?')}, strict={load_strict})",
            flush=True,
        )
        if not args.eval_only:
            with open(out_dir / "init_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "path": str(ckpt_path.resolve()),
                        "epoch": ckpt.get("epoch"),
                        "strict": load_strict,
                    },
                    f,
                    indent=2,
                )

    if args.eval_only:
        if ckpt_path is None:
            raise FileNotFoundError(
                "eval-only requires --checkpoint, paths.init_checkpoint, or existing output_dir/best_model.pth"
            )
        finalize_eval_export(
            model, dl_val, dl_test, out_dir, cfg, device, use_amp, w_prob, w_reg_inf
        )
        return

    core = unwrap_model(model)
    backbone_params = list(get_encoder_backbone(model).parameters())
    backbone_ids = {id(p) for p in backbone_params}
    trainable_backbone = [p for p in backbone_params if p.requires_grad]
    rest_params = [p for p in core.parameters() if p.requires_grad and id(p) not in backbone_ids]

    if trainable_backbone:
        opt = torch.optim.AdamW(
            [
                {"params": trainable_backbone, "lr": float(cfg["train"]["backbone_lr"])},
                {"params": rest_params, "lr": float(cfg["train"]["head_lr"])},
            ],
            weight_decay=float(cfg["train"]["weight_decay"]),
        )
    else:
        opt = torch.optim.AdamW(
            rest_params,
            lr=float(cfg["train"]["head_lr"]),
            weight_decay=float(cfg["train"]["weight_decay"]),
        )

    loss_fn = ClinicalLQFTotalLoss(
        num_classes=int(cfg["data"]["num_classes"]),
        w_cls=float(cfg["loss"]["w_cls"]),
        w_ord=float(cfg["loss"]["w_ord"]),
        w_reg=float(cfg["loss"]["w_reg"]),
        w_relation=float(cfg["loss"]["w_relation"]),
        w_burden=float(cfg["loss"]["w_burden"]),
        relation_tau=float(cfg["loss"].get("relation_tau", 0.7)),
        ce_class_weights=cw,
    ).to(device)

    scaler = GradScaler(enabled=use_amp)

    use_sampler = bool(cfg["train"].get("use_weighted_sampler", False))
    if use_sampler:
        sw = per_sample_weights(tlabs, int(cfg["data"]["num_classes"]), mode="inverse")
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        dl_train = DataLoader(ds_train, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=True)
        print("WeightedRandomSampler enabled (inverse training class frequency)", flush=True)
    else:
        dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)

    epochs = int(cfg["train"]["epochs"])
    warmup = int(cfg["train"]["warmup_epochs"])
    accum = max(1, int(cfg["train"]["grad_accum_steps"]))
    clip = float(cfg["train"]["grad_clip"])

    keep_top_k = int(cfg["train"].get("keep_top_k_checkpoints", 2))
    ckpt_mgr = TopKCheckpointManager(out_dir, k=keep_top_k)
    log_f = open(out_dir / "training_log.txt", "w", encoding="utf-8")

    for epoch in range(epochs):
        fac = lr_factor(epoch, warmup, epochs)
        if len(opt.param_groups) == 2:
            opt.param_groups[0]["lr"] = float(cfg["train"]["backbone_lr"]) * fac
            opt.param_groups[1]["lr"] = float(cfg["train"]["head_lr"]) * fac
        else:
            opt.param_groups[0]["lr"] = float(cfg["train"]["head_lr"]) * fac

        model.train()
        running = 0.0
        part_sums = {k: 0.0 for k in ("L_cls", "L_ord", "L_reg", "L_rel", "L_burden")}
        n_batches = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(dl_train, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
            x = batch["x_rgb"].to(device, non_blocking=True)
            p = batch["p_lesion"].to(device, non_blocking=True)
            s = batch["s_lesion"].to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                out = model(x, p, s)
                loss, parts = loss_fn(out, batch)
                aux_w = float(cfg["model"].get("aux_head_weight", 0.0))
                if aux_w > 0 and out.get("logits_aux1") is not None:
                    lab = batch["label"].to(device)
                    loss = loss + aux_w * (
                        F.cross_entropy(out["logits_aux1"], lab)
                        + F.cross_entropy(out["logits_aux3"], lab)
                    )
                group_size = min(accum, len(dl_train) - (step // accum) * accum)
                loss = loss / group_size
            scaler.scale(loss).backward()
            running += loss.item() * group_size
            for k in part_sums:
                if k in parts:
                    part_sums[k] += float(parts[k].detach().float().mean().item())
            n_batches += 1
            if (step + 1) % accum == 0 or (step + 1) == len(dl_train):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{running / max(1, step + 1):.4f}")

        _, _, _, _, _, val_qwk_argmax = run_epoch_eval(
            model, dl_val, device, use_amp, w_prob, w_reg_inf
        )
        msg = f"epoch {epoch} train_loss {running/max(1,len(dl_train)):.4f} val_qwk_argmax {val_qwk_argmax:.4f}\n"
        log_f.write(msg)
        log_f.flush()
        print(msg.strip())
        # Log each loss term before and after weighting.
        if n_batches > 0:
            raw = {k: part_sums[k] / n_batches for k in part_sums}
            wmap = {
                "L_cls": float(loss_fn.w_cls),
                "L_ord": float(loss_fn.w_ord),
                "L_reg": float(loss_fn.w_reg),
                "L_rel": float(loss_fn.w_relation),
                "L_burden": float(loss_fn.w_burden),
            }
            weighted = {k: wmap[k] * raw[k] for k in raw}
            terms_path = out_dir / "loss_terms_by_epoch.jsonl"
            with terms_path.open("a", encoding="utf-8") as tf:
                tf.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "raw": raw,
                            "weighted": weighted,
                            "weights": wmap,
                            "train_loss": running / max(1, len(dl_train)),
                            "val_qwk_argmax": float(val_qwk_argmax),
                        }
                    )
                    + "\n"
                )

        if ckpt_mgr.maybe_save(val_qwk_argmax, epoch, model, cfg):
            print(f"  checkpoint saved (val QWK={val_qwk_argmax:.4f})", flush=True)

    log_f.close()

    best_path = out_dir / "best_model.pth"
    if not ckpt_mgr.records:
        raise RuntimeError("No finite validation QWK checkpoint was saved")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    load_model_checkpoint(model, ckpt, strict=True)
    finalize_eval_export(
        model, dl_val, dl_test, out_dir, cfg, device, use_amp, w_prob, w_reg_inf
    )

    print("done. output:", out_dir)


if __name__ == "__main__":
    main()
