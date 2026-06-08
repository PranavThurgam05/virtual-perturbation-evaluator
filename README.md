# Virtual Perturbation Evaluator

This project trains simple neural predictors for Arc Virtual Cell Challenge-style
gene perturbation response modeling. The implemented task is a pseudobulk
perturbation-delta benchmark:

1. preprocess single-cell AnnData into perturbation-level means and deltas
2. train MLP, Mamba, or Transformer models to predict deltas for target genes
3. generate post-perturbation expression predictions
4. evaluate with fast local proxy metrics and, for final reporting, Arc's
   `cell-eval` metrics

The current model does not yet condition on individual cell state. It predicts
one pseudobulk response per perturbation target from target-gene features and a
global control expression profile.

## Setup

Use the locked environment when possible:

```powershell
uv sync --locked
```

The lockfile installs the project with Python 3.11 and CUDA PyTorch. If you only
need official evaluation, install Arc's evaluator inside the same environment:

```powershell
uv pip install cell-eval
```

## Data Layout

Expected paths are configured in `configs/data.yaml`:

```text
data/raw/arc/adata_Training.h5ad
data/processed/perturbation_delta_dataset.npz
data/splits/gene_split_seed42.json
data/raw/test/test.csv
```

The raw and processed data files are intentionally ignored by git.

## Preprocess

```powershell
uv run python scripts/preprocess.py --config configs/data.yaml
```

This performs QC, normalization, HVG selection, scVI batch correction,
pseudobulking, delta construction, target-gene feature extraction, and train/val
gene splitting.

## Train

```powershell
uv run python scripts/train.py --config configs/mlp.yaml
uv run python scripts/train.py --config configs/mamba.yaml
uv run python scripts/train.py --config configs/transformer.yaml
```

By default WandB is disabled in all configs. Enable it explicitly in the config
if you want hosted experiment tracking.

The Transformer config is intentionally conservative because full self-attention
over thousands of genes is memory-heavy. Increase `batch_size`, `hidden_size`, or
`num_layers` only after the small config runs on your GPU.

## Predict

Generate validation-split predictions from a trained checkpoint:

```powershell
uv run python scripts/predict.py --config configs/mlp.yaml
```

Generate predictions for a metadata CSV, writing both NPZ and H5AD outputs:

```powershell
uv run python scripts/predict.py `
  --config configs/mlp.yaml `
  --metadata data/raw/test/test.csv `
  --output-npz outputs/predictions/mlp_test_predictions.npz `
  --output-h5ad outputs/predictions/mlp_test_predictions.h5ad
```

The H5AD writer repeats each pseudobulk prediction into synthetic cells and
includes repeated control rows so downstream differential-expression metrics can
compare perturbations to controls.

## Evaluate

Training logs fast local proxy metrics from `src/vcell/metrics.py`. These are
for debugging and model selection only.

For final reporting, use Arc's `cell-eval` evaluator. The wrapper writes
`results.csv` and `agg_results.csv` using Arc's naming convention:

```powershell
uv pip install cell-eval

uv run python scripts/evaluate_official.py `
  --config configs/data.yaml `
  --pred-h5ad outputs/predictions/mlp_test_predictions.h5ad `
  --real-h5ad data/raw/test/ground_truth.h5ad `
  --output-dir outputs/cell_eval/mlp
```

Replace `ground_truth.h5ad` with the held-out validation or benchmark file that
contains the true post-perturbation cells.

## Smoke Test

After installing dependencies, run the lightweight synthetic check:

```powershell
uv run python scripts/smoke_test.py
```

To also instantiate tiny Transformer and Mamba models:

```powershell
uv run python scripts/smoke_test.py --include-sequence-models
```

## Suggested Final Report Outputs

- validation metrics JSON from each model in `outputs/metrics/`
- official `cell-eval` aggregate metrics for the best checkpoint
- runtime and parameter-count comparison from the saved metrics JSON files
- PCA/UMAP diagnostic plots from `scripts/investigate_pca.py`
- a limitations note explaining that the current implementation is pseudobulk,
  not per-cell contextual prediction
