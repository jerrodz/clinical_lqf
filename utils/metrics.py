from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def evaluate_preds(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> dict:
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class_recall": recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def score_continuous(logits: np.ndarray, score_reg: np.ndarray, w_prob: float = 0.7, w_reg: float = 0.3) -> np.ndarray:
    prob = softmax(logits, axis=1)
    score_prob = (prob * np.arange(logits.shape[1], dtype=np.float64)).sum(axis=1)
    return w_prob * score_prob + w_reg * score_reg


def discretize_scores(scores: np.ndarray, t: np.ndarray) -> np.ndarray:
    t1, t2, t3, t4 = t
    p = np.zeros(len(scores), dtype=np.int64)
    p[scores >= t4] = 4
    p[(scores >= t3) & (scores < t4)] = 3
    p[(scores >= t2) & (scores < t3)] = 2
    p[(scores >= t1) & (scores < t2)] = 1
    p[scores < t1] = 0
    return p


def _threshold_objective_score(
    qwk: float,
    acc: float,
    objective: str = "mean",
    qwk_weight: float = 0.5,
    acc_weight: float = 0.5,
) -> float:
    """Scalar used to rank thresholds on the search pool (val / train+val only)."""
    obj = str(objective).lower().strip()
    qw = float(qwk_weight)
    aw = float(acc_weight)
    if obj in ("qwk", "kappa"):
        return float(qwk)
    if obj in ("acc", "accuracy"):
        return float(acc)
    if obj in ("min", "min_both", "minmax"):
        return float(min(qwk, acc))
    if obj in ("product", "prod"):
        return float(qwk * acc)
    if obj in ("harmonic", "hmean"):
        s = qwk + acc
        return float(2.0 * qwk * acc / s) if s > 1e-12 else 0.0
    # default: weighted mean — both metrics high
    wsum = qw + aw
    if wsum <= 1e-12:
        return 0.5 * (qwk + acc)
    return float((qw * qwk + aw * acc) / wsum)


def search_thresholds_qwk(
    scores: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 4000,
    seed: int = 42,
    min_pred_frac: list[float] | tuple[float, ...] | None = None,
    min_gap: float = 0.2,
    objective: str = "mean",
    qwk_weight: float = 0.5,
    acc_weight: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Threshold search maximizing QWK+Acc (configurable), with coverage / min-gap constraints.

    Default ``objective=mean`` ranks by ``(qwk_weight*QWK + acc_weight*Acc) / (qw+aw)``,
    with tie-break: higher ``min(QWK,Acc)``, then higher QWK, then higher Acc.
    Other objectives: ``qwk`` | ``acc`` | ``min_both`` | ``harmonic`` | ``product``.

    Returns ``(best_thresholds, pool_qwk_at_pick)``. Use ``search_thresholds_detail``
    if you also need Acc / composite score at the pick.
    """
    best_t, detail = search_thresholds_detail(
        scores,
        labels,
        n_samples=n_samples,
        seed=seed,
        min_pred_frac=min_pred_frac,
        min_gap=min_gap,
        objective=objective,
        qwk_weight=qwk_weight,
        acc_weight=acc_weight,
    )
    return best_t, float(detail["qwk"])


def search_thresholds_detail(
    scores: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 4000,
    seed: int = 42,
    min_pred_frac: list[float] | tuple[float, ...] | None = None,
    min_gap: float = 0.2,
    objective: str = "mean",
    qwk_weight: float = 0.5,
    acc_weight: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Same search as ``search_thresholds_qwk``, but returns Acc / score at pick."""
    rng = np.random.RandomState(seed)
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        hi = lo + 1e-3
    n_cls = int(labels.max()) + 1 if len(labels) else 5
    true_counts = np.bincount(labels.astype(np.int64), minlength=n_cls).astype(np.float64)
    frac = None
    if min_pred_frac is not None:
        frac = np.asarray(min_pred_frac, dtype=np.float64)
        if frac.shape[0] < n_cls:
            frac = np.pad(frac, (0, n_cls - frac.shape[0]))
        frac = frac[:n_cls]

    candidates: list[np.ndarray] = []
    for qs in (
        [20, 40, 60, 80],
        [15, 35, 65, 85],
        [10, 30, 70, 90],
        [25, 45, 70, 88],
        [12, 28, 55, 82],
        [18, 38, 62, 86],
    ):
        candidates.append(np.percentile(scores, qs).astype(np.float64))
    means = []
    for g in range(n_cls):
        s = scores[labels == g]
        means.append(float(s.mean()) if len(s) else float(g))
    if len(means) >= 5:
        mid = np.array([(means[i] + means[i + 1]) * 0.5 for i in range(4)], dtype=np.float64)
        for scale in (0.9, 1.0, 1.1):
            candidates.append(np.sort(mid * scale))
    for _ in range(n_samples):
        candidates.append(np.sort(rng.uniform(lo, hi, size=4)))

    best_score = -1.0
    best_qwk = -1.0
    best_acc = -1.0
    best_min = -1.0
    best_t = np.array([0.25, 1.0, 2.0, 3.0], dtype=np.float64)
    for t in candidates:
        t = np.sort(np.asarray(t, dtype=np.float64))
        if min_gap > 0 and np.min(np.diff(t)) < min_gap:
            continue
        pred = discretize_scores(scores, t)
        if frac is not None:
            pred_counts = np.bincount(pred.astype(np.int64), minlength=n_cls).astype(np.float64)
            ok = True
            for c in range(n_cls):
                need = frac[c] * true_counts[c]
                if need > 0 and pred_counts[c] + 1e-9 < need:
                    ok = False
                    break
            if not ok:
                continue
        qwk = float(cohen_kappa_score(labels, pred, weights="quadratic"))
        acc = float(accuracy_score(labels, pred))
        score = _threshold_objective_score(qwk, acc, objective, qwk_weight, acc_weight)
        both_min = min(qwk, acc)
        better = False
        if score > best_score + 1e-15:
            better = True
        elif abs(score - best_score) <= 1e-15:
            if both_min > best_min + 1e-15:
                better = True
            elif abs(both_min - best_min) <= 1e-15 and (
                qwk > best_qwk + 1e-15 or (abs(qwk - best_qwk) <= 1e-15 and acc > best_acc + 1e-15)
            ):
                better = True
        if better:
            best_score = score
            best_qwk = qwk
            best_acc = acc
            best_min = both_min
            best_t = t.copy()
    return best_t, {
        "qwk": best_qwk,
        "acc": best_acc,
        "score": best_score,
        "objective": str(objective).lower().strip(),
        "qwk_weight": float(qwk_weight),
        "acc_weight": float(acc_weight),
    }
