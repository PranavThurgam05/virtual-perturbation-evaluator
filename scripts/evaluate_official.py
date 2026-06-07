import argparse
from pathlib import Path


from vcell.utils import load_yaml


def save_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "to_csv"):
        result.to_csv(path, index=False)
    else:
        path.write_text(str(result))


def main():
    parser = argparse.ArgumentParser(
        description="Run Arc cell-eval metrics on prediction and ground-truth H5AD files."
    )
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--pred-h5ad", type=str, required=True)
    parser.add_argument("--real-h5ad", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/cell_eval")
    parser.add_argument("--control-label", type=str, default=None)
    parser.add_argument("--pert-col", type=str, default=None)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument(
        "--baseline-agg",
        type=str,
        default=None,
        help="Optional baseline agg_results.csv for normalized cell-eval scoring.",
    )
    args = parser.parse_args()

    try:
        import anndata as ad
        from cell_eval import MetricsEvaluator, score_agg_metrics
    except ImportError as exc:
        raise ImportError(
            "Official evaluation requires cell-eval and anndata. "
            "Install them in the project environment, for example: "
            "`uv pip install cell-eval`."
        ) from exc

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pert_col = args.pert_col or data_cfg.get("target_col", "target_gene")
    control_label = args.control_label or data_cfg.get("control_label", "non-targeting")

    adata_pred = ad.read_h5ad(args.pred_h5ad)
    adata_real = ad.read_h5ad(args.real_h5ad)

    kwargs = {
        "adata_pred": adata_pred,
        "adata_real": adata_real,
        "control_pert": control_label,
        "pert_col": pert_col,
        "num_threads": int(args.num_threads),
        "profile": args.profile,
    }
    try:
        evaluator = MetricsEvaluator(**kwargs)
    except TypeError:
        kwargs.pop("profile", None)
        evaluator = MetricsEvaluator(**kwargs)

    results = evaluator.compute()
    if isinstance(results, tuple):
        per_pert, aggregate = results
    else:
        per_pert, aggregate = results, None

    save_result(per_pert, output_dir / "results.csv")
    if aggregate is not None:
        agg_path = output_dir / "agg_results.csv"
        save_result(aggregate, agg_path)
        if args.baseline_agg:
            score_agg_metrics(
                results_user=str(agg_path),
                results_base=args.baseline_agg,
                output=str(output_dir / "score.csv"),
            )

    print(f"Saved cell-eval results -> {output_dir}")


if __name__ == "__main__":
    main()
