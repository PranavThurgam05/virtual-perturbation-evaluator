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

    ranks = np.asarray(ranks, dtype=np.float64)
    n = distances.shape[0]

    reciprocal_ranks = 1.0 / ranks
    top1 = np.mean(ranks == 1)

    # Official-style VCC Perturbation Discrimination Score.
    # Per-sample normalized rank in [0, 1] (0 = best, ~0.5 = chance, 1 = worst),
    # then map to a score where perfect = 1, random = 0, worst = -1.
    # Unlike MRR this scales linearly with rank, so a model that places the
    # correct perturbation in the top few percent scores near 1 instead of
    # looking deceptively bad. Prefer this for cross-run comparison.
    norm_rank = (ranks - 1.0) / max(n - 1, 1)
    pds_norm = float(1.0 - 2.0 * np.mean(norm_rank))

    return {
        "pds_norm": pds_norm,
        "pds_mrr": float(np.mean(reciprocal_ranks)),
        "pds_top1": float(top1),
        "mean_rank": float(np.mean(ranks)),
    }


def simplified_des_topk(pred_deltas, true_deltas, k=100):
    """
    Simplified Differential Expression Score proxy.

    NOTE: This is *not* the official VCC DES, which runs a Wilcoxon test of
    perturbed-vs-control cells to obtain a significance-determined (variable
    size, signed) DE-gene set. That requires per-cell data not available here.
    This proxy uses the top-k genes by |delta| and additionally checks that the
    regulation *direction* (up/down) agrees, which the magnitude-only version
    ignored. Chance baseline for the overlap is ~k / n_genes, so interpret
    absolute values against that floor, not against 1.0.
    """
    overlaps = []
    signed_overlaps = []
    for pred, true in zip(pred_deltas, true_deltas):
        pred_top = np.argsort(np.abs(pred))[-k:]
        true_top = np.argsort(np.abs(true))[-k:]
        pred_set, true_set = set(pred_top), set(true_top)
        common = pred_set & true_set
        overlaps.append(len(common) / k)
        # Of the shared top-k genes, fraction whose sign also matches.
        if common:
            agree = sum(np.sign(pred[g]) == np.sign(true[g]) for g in common)
            signed_overlaps.append(agree / k)
        else:
            signed_overlaps.append(0.0)

    return {
        f"des_top{k}_overlap": float(np.mean(overlaps)),
        f"des_top{k}_signed_overlap": float(np.mean(signed_overlaps)),
    }


def prediction_collapse_diagnostic(pred_deltas, true_deltas):
    """
    Detects mean-collapse: a model that ignores perturbation identity and
    predicts (nearly) the same delta for every perturbation gets good MAE but
    chance-level discrimination. We compare the spread of predictions across
    perturbations to the spread of the ground truth.

    pred_dispersion_ratio ~ 0  -> collapsed to a constant (bad)
    pred_dispersion_ratio ~ 1  -> healthy per-gene spread
    """
    pred_spread = float(np.mean(np.std(pred_deltas, axis=0)))
    true_spread = float(np.mean(np.std(true_deltas, axis=0)))
    return {
        "pred_spread": pred_spread,
        "true_spread": true_spread,
        "pred_dispersion_ratio": pred_spread / (true_spread + 1e-8),
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
    out.update(prediction_collapse_diagnostic(pred_deltas, true_deltas))

    if pred_means is not None and true_means is not None:
        out["pseudobulk_mae"] = mae(pred_means, true_means)

    return out