import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from vcell.data import PerturbationDeltaDataset, load_processed_npz
from vcell.metrics import evaluate_delta_predictions
from vcell.models.factory import build_model
from vcell.utils import (
    count_parameters,
    ensure_parent,
    get_device,
    load_yaml,
    save_json,
    set_seed,
)


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def weighted_huber_loss(pred, target):
    """
    Weight larger true deltas more so the model does not ignore DE-like genes.
    """
    base = torch.nn.functional.huber_loss(pred, target, reduction="none", delta=1.0)
    weights = 1.0 + 4.0 * (
        target.abs() > target.abs().quantile(0.95, dim=1, keepdim=True)
    ).float()
    return (base * weights).mean()


@torch.no_grad()
def predict_all(model, loader, device, control_mean):
    model.eval()
    pred_deltas, true_deltas, true_means = [], [], []

    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch)
        pred_deltas.append(pred.detach().cpu().numpy())
        true_deltas.append(batch["delta"].detach().cpu().numpy())
        true_means.append(batch["true_mean"].detach().cpu().numpy())

    pred_deltas = np.concatenate(pred_deltas, axis=0)
    true_deltas = np.concatenate(true_deltas, axis=0)
    true_means  = np.concatenate(true_means,  axis=0)
    pred_means  = control_mean[None, :] + pred_deltas

    return pred_deltas, true_deltas, pred_means, true_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg  = load_yaml(args.config)
    tcfg = cfg["training"]
    wcfg = cfg.get("wandb", {})

    # Default wandb to False — avoids crashing when package is absent or
    # key is missing from config entirely.
    use_wandb = bool(wcfg.get("enabled", False))

    set_seed(int(tcfg.get("seed", 42)))
    device = get_device()
    print(f"Using device: {device}")

    processed = load_processed_npz(cfg["data"]["processed_npz"])

    train_ds = PerturbationDeltaDataset(
        processed, split_json=cfg["data"]["split_json"], split="train"
    )
    val_ds = PerturbationDeltaDataset(
        processed, split_json=cfg["data"]["split_json"], split="val"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    model = build_model(cfg, processed, device)
    print(f"Model type:     {cfg['model_type']}")
    print(f"Train samples:  {len(train_ds)}")
    print(f"Val samples:    {len(val_ds)}")
    print(f"Parameters:     {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    # Cosine LR schedule: decays lr smoothly to near-zero by the final epoch.
    # This lets the model converge better in later epochs without manual tuning.
    n_epochs  = int(tcfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=float(tcfg["lr"]) * 0.01
    )

    # ── wandb setup ──────────────────────────────────────────────────────────
    wandb_run = None
    if use_wandb:
        if wandb is None:
            raise ImportError(
                "wandb is enabled in config but not installed. "
                "Run: pip install wandb"
            )
        wandb_run = wandb.init(
            project=wcfg.get("project", "virtual-perturbation-evaluator"),
            entity=wcfg.get("entity"),
            name=wcfg.get("run_name"),
            tags=wcfg.get("tags"),
            config=cfg,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    # ── training state ───────────────────────────────────────────────────────
    best_val     = float("inf")
    best_metrics = None
    patience     = int(tcfg.get("patience", 10))
    bad_epochs   = 0

    checkpoint_path = Path(tcfg["checkpoint_path"])
    ensure_parent(checkpoint_path)

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()

        # ── train ─────────────────────────────────────────────────────────
        model.train()
        losses = []

        for batch in train_loader:
            batch = move_batch(batch, device)

            optimizer.zero_grad()          # FIX: clear grads before forward pass
            pred = model(batch)
            loss = weighted_huber_loss(pred, batch["delta"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        scheduler.step()

        # ── validate ──────────────────────────────────────────────────────
        pred_deltas, true_deltas, pred_means, true_means = predict_all(
            model, val_loader, device, processed.control_mean
        )
        metrics = evaluate_delta_predictions(
            pred_deltas=pred_deltas,
            true_deltas=true_deltas,
            pred_means=pred_means,
            true_means=true_means,
        )

        val_loss       = metrics["delta_mae"]
        train_loss     = float(np.mean(losses))
        epoch_seconds  = time.time() - epoch_start
        current_lr     = scheduler.get_last_lr()[0]

        # ETA based on time so far, capped at patience epochs remaining
        avg_epoch_s  = (time.time() - start_time) / epoch
        epochs_left  = min(n_epochs - epoch, patience - bad_epochs)
        eta_seconds  = max(0.0, avg_epoch_s * epochs_left)

        print(
            f"epoch {epoch:03d} | "
            f"train={train_loss:.5f} | "
            f"val_mae={metrics['delta_mae']:.5f} | "
            f"pds_top1={metrics['pds_top1']:.3f} | "
            f"des100={metrics['des_top100_overlap']:.3f} | "
            f"lr={current_lr:.2e} | "
            f"epoch_s={epoch_seconds:.1f} | "
            f"eta_m={eta_seconds / 60.0:.1f}"
        )

        if wandb_run is not None:
            epoch_log = {
                "epoch":               epoch,
                "train/loss":          train_loss,
                "train/lr":            current_lr,
                "train/epoch_seconds": float(epoch_seconds),
                "train/eta_seconds":   float(eta_seconds),
            }
            for k, v in metrics.items():
                epoch_log[f"val/{k}"] = float(v)
            wandb_run.log(epoch_log)

        # ── checkpoint ────────────────────────────────────────────────────
        if val_loss < best_val:
            best_val     = val_loss
            best_metrics = metrics
            bad_epochs   = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config":      cfg,
                    "gene_names":  processed.gene_names,
                },
                checkpoint_path,
            )
            print(f"  saved checkpoint → {checkpoint_path}")

            if wandb_run is not None:
                wandb_run.log({
                    "epoch":           epoch,
                    "best/epoch":      epoch,
                    "best/delta_mae":  float(best_val),
                })
        else:
            bad_epochs += 1

        # ── early stopping ────────────────────────────────────────────────
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            if wandb_run is not None:
                wandb_run.log({"epoch": epoch, "train/early_stop_epoch": epoch})
            break

    # ── final metrics ────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    if best_metrics is None:
        # Training ended without a single improvement — flag clearly rather
        # than silently saving an empty metrics file.
        raise RuntimeError(
            "No checkpoint was saved — val loss never improved. "
            "Check your data, model, and learning rate."
        )

    best_metrics["training_seconds"] = elapsed
    best_metrics["num_parameters"]   = count_parameters(model)
    best_metrics["model_type"]       = cfg["model_type"]

    metrics_path = tcfg["metrics_path"]
    save_json(best_metrics, metrics_path)
    print(f"Saved metrics → {metrics_path}")
    print("Best metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v}")

    if wandb_run is not None:
        wandb_run.summary["training_seconds"] = float(best_metrics["training_seconds"])
        wandb_run.summary["num_parameters"]   = int(best_metrics["num_parameters"])
        wandb_run.summary["model_type"]       = best_metrics["model_type"]
        for k, v in best_metrics.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                wandb_run.summary[f"best/{k}"] = float(v)
        wandb.finish()


if __name__ == "__main__":
    main()
