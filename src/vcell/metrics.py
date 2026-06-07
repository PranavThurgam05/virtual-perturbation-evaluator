"""
Fast validation metrics for local debugging.

These metrics are intentionally lightweight proxies. Use scripts/evaluate_official.py
with Arc's cell-eval package for final VCC-style DES/PDS/MAE reporting.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def cosine_similarity_mean(pred, true):
    pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    true_norm = true / (np.linalg.norm(true, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(pred_norm * true_norm, axis=1)))


def pds_l1_rank_score(pred_deltas, true_deltas):
    """
    Simplified PDS-style metric.

    For each predicted perturbation delta, compare against all true heldout deltas.
    Score is mean reciprocal rank of the correct perturbation.
    Higher is better.
    """
    distances = cdist(pred_deltas, true_deltas, metric="cityblock")
    ranks = []
    for i in range(distances.shape[0]):
        order = np.argsort(distances[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)

    reciprocal_ranks = [1.0 / r for r in ranks]
    top1 = np.mean([r == 1 for r in ranks])
    return {
        "pds_mrr": float(np.mean(reciprocal_ranks)),
        "pds_top1": float(top1),
        "mean_rank": float(np.mean(ranks)),
    }


def simplified_des_topk(pred_deltas, true_deltas, k=100):
    """
    Simplified Differential Expression Score:
    overlap between top-k absolute predicted delta genes and top-k true delta genes.
    """
    overlaps = []
    for pred, true in zip(pred_deltas, true_deltas):
        pred_top = set(np.argsort(np.abs(pred))[-k:])
        true_top = set(np.argsort(np.abs(true))[-k:])
        overlaps.append(len(pred_top & true_top) / k)

    return {
        f"des_top{k}_overlap": float(np.mean(overlaps)),
    }


def delta_spearman(pred_deltas, true_deltas):
    vals = []
    for pred, true in zip(pred_deltas, true_deltas):
        r, _ = spearmanr(pred, true)
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_delta_predictions(pred_deltas, true_deltas, pred_means=None, true_means=None):
    out = {
        "delta_mae": mae(pred_deltas, true_deltas),
        "delta_cosine": cosine_similarity_mean(pred_deltas, true_deltas),
        "delta_spearman": delta_spearman(pred_deltas, true_deltas),
    }
    out.update(pds_l1_rank_score(pred_deltas, true_deltas))
    out.update(simplified_des_topk(pred_deltas, true_deltas, k=50))
    out.update(simplified_des_topk(pred_deltas, true_deltas, k=100))
    out.update(simplified_des_topk(pred_deltas, true_deltas, k=200))

    if pred_means is not None and true_means is not None:
        out["pseudobulk_mae"] = mae(pred_means, true_means)

    return out
