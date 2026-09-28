# Classification metrics with 95% percentile-bootstrap CIs. Positive class = hallucinated.

from typing import Dict, Optional, Sequence

import numpy as np

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def roc_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    # Mann-Whitney U with average ranks for ties.
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def average_precision(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[ss[1:] != ss[:-1], True]  # one point per tied score
    precision = tp[last] / (tp[last] + fp[last])
    recall = tp[last] / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def point_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> Dict[str, Optional[float]]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    has_pos, has_neg = bool((y == 1).any()), bool((y == 0).any())
    precision, recall, specificity = _div(tp, tp + fp), _div(tp, tp + fn), _div(tn, tn + fp)
    return {
        "accuracy": float((pred == y).mean()) if len(y) else None,
        "balanced_accuracy": (recall + specificity) / 2 if has_pos and has_neg else None,
        "precision": precision if has_pos else None,
        "recall": recall if has_pos else None,
        "f1": _div(2 * precision * recall, precision + recall) if has_pos else None,
        "roc_auc": roc_auc(y, score),
        "pr_auc": average_precision(y, score),
    }


def evaluate(y_true: Sequence[bool], y_pred: Sequence[bool], y_score: Sequence[float],
             n_bootstrap: int = 1000, seed: int = 0) -> Dict[str, dict]:
    y, p, s = np.asarray(y_true, int), np.asarray(y_pred, int), np.asarray(y_score, float)
    point = point_metrics(y, p, s)
    out = {k: {"value": v, "ci_low": None, "ci_high": None} for k, v in point.items()}
    if n_bootstrap and len(y) > 1:
        rng = np.random.default_rng(seed)
        boots = {k: [] for k in point}
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y), len(y))
            for k, v in point_metrics(y[idx], p[idx], s[idx]).items():
                if v is not None:
                    boots[k].append(v)
        for k, vals in boots.items():
            if point[k] is not None and len(vals) >= 0.9 * n_bootstrap:
                out[k]["ci_low"], out[k]["ci_high"] = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
    out["n"] = {"value": int(len(y)), "ci_low": None, "ci_high": None}
    out["n_hallucinated"] = {"value": int(y.sum()), "ci_low": None, "ci_high": None}
    return out


def paired_diff(y_true, pred_a, score_a, pred_b, score_b, metric: str,
                n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """metric(A) - metric(B) on the same samples, with a paired-bootstrap 95% CI."""
    y = np.asarray(y_true, int)
    pa, sa = np.asarray(pred_a, int), np.asarray(score_a, float)
    pb, sb = np.asarray(pred_b, int), np.asarray(score_b, float)

    def diff(idx):
        a = point_metrics(y[idx], pa[idx], sa[idx])[metric]
        b = point_metrics(y[idx], pb[idx], sb[idx])[metric]
        return None if a is None or b is None else a - b

    full = diff(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    boots = [d for d in (diff(rng.integers(0, len(y), len(y))) for _ in range(n_bootstrap))
             if d is not None]
    lo = hi = None
    if boots and len(boots) >= 0.9 * n_bootstrap:
        lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return {"metric": metric, "diff": full, "ci_low": lo, "ci_high": hi, "n": int(len(y))}
