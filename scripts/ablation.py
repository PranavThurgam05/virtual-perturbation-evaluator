"""
Transformer input-ablation harness.

Trains the transformer several times, each with one input stream disabled, and
prints a comparison table. Use it to answer: which inputs actually drive the
prediction? If removing the perturbation-specific inputs (target / indicator)
barely changes pds_norm, the model is collapsing toward the mean delta.

Usage:
    PYTHONPATH=src python scripts/ablation.py --config configs/transformer.yaml
    # or point at the synthetic dataset for a fast smoke run:
    PYTHONPATH=src python scripts/ablation.py --config configs/transformer.yaml \
        --processed-npz data/processed/synthetic_dataset.npz \
        --split-json data/splits/synthetic_split.json --epochs 5
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train import run_training
from vcell.utils import load_yaml


# name -> model flag overrides. "full" disables nothing.
ABLATIONS = {
    "full":          {},
    "no_gene":       {"use_gene": False},
    "no_expr":       {"use_expr": False},
    "no_target":     {"use_target": False},
    "no_indicator":  {"use_indicator": False},
}

REPORT_KEYS = ["pds_norm", "delta_mae", "delta_cosine",
               "des_top100_overlap", "pred_dispersion_ratio"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/transformer.yaml")
    p.add_argument("--processed-npz", type=str, default=None)
    p.add_argument("--split-json", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="outputs/ablation")
    args = p.parse_args()

    base = load_yaml(args.config)
    if base.get("model_type") != "transformer":
        raise ValueError("ablation.py expects a transformer config (model_type: transformer)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, flags in ABLATIONS.items():
        cfg = copy.deepcopy(base)
        cfg.setdefault("model", {}).update(flags)
        cfg.setdefault("data", {})
        if args.processed_npz:
            cfg["data"]["processed_npz"] = args.processed_npz
        if args.split_json:
            cfg["data"]["split_json"] = args.split_json
        if args.epochs is not None:
            cfg["training"]["epochs"] = args.epochs
        # Isolate artifacts per run and never start wandb from the harness.
        cfg["training"]["checkpoint_path"] = str(out_dir / f"{name}.pt")
        cfg["training"]["metrics_path"] = str(out_dir / f"{name}_metrics.json")
        cfg.setdefault("wandb", {})["enabled"] = False

        print(f"\n{'='*70}\n  ABLATION: {name}  (flags: {flags or 'none'})\n{'='*70}")
        results[name] = run_training(cfg)

    # ── comparison table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  ABLATION SUMMARY\n{'='*70}")
    header = f"{'variant':<14}" + "".join(f"{k:>22}" for k in REPORT_KEYS)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = f"{name:<14}" + "".join(f"{m.get(k, float('nan')):>22.4f}" for k in REPORT_KEYS)
        print(row)

    full = results.get("full", {})
    print("\nReading the table:")
    print("  - Big pds_norm drop vs 'full' when a stream is removed => that input matters.")
    print("  - If 'no_target' and 'no_indicator' are close to 'full', the model is")
    print("    ignoring perturbation identity (mean-collapse).")
    if full:
        print(f"  - full pds_norm = {full.get('pds_norm', float('nan')):.4f}, "
              f"dispersion = {full.get('pred_dispersion_ratio', float('nan')):.4f}")


if __name__ == "__main__":
    main()
