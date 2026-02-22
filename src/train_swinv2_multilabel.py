from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import MultiLabelImageList
from swinv2_multilabel import (
    build_model,
    build_transforms,
    load_config,
    load_split,
    multilabel_counts_from_preds,
    multilabel_metrics_from_counts,
    resolve_labels,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synapse_swinv2_multilabel.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    task_type = (cfg.get("task_type", "multilabel") or "multilabel").strip().lower()
    if task_type != "multilabel":
        raise ValueError("train_swinv2_multilabel.py only supports task_type=multilabel")

    seed = int(cfg["seed"])
    torch.manual_seed(seed)

    output_dir = Path(cfg["output_dir"])
    split_dir = output_dir / "splits"
    if not (split_dir / "train.json").exists():
        raise FileNotFoundError(
            f"multilabel training expects precomputed splits at {split_dir}. "
            "Run scripts/prepare_synapse_classification.py first."
        )

    train_items, val_items, _ = load_split(split_dir)
    labels = resolve_labels(output_dir, train_items)
    num_classes = len(labels)

    tf_train, tf_eval = build_transforms(cfg, train_items)

    train_ds = MultiLabelImageList(train_items, transform=tf_train)
    val_ds = MultiLabelImageList(val_items, transform=tf_eval)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_classes=num_classes).to(device)

    pos_counts = torch.zeros(num_classes, dtype=torch.float32)
    for item in train_items:
        y = item["labels"] if isinstance(item, dict) else item[1]
        pos_counts += torch.tensor(y, dtype=torch.float32)
    n = max(float(len(train_items)), 1.0)
    neg_counts = n - pos_counts
    pos_weight = neg_counts / torch.clamp(pos_counts, min=1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    opt_cfg = cfg.get("optimizer", {}) or {}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        betas=(float(opt_cfg.get("beta1", 0.9)), float(opt_cfg.get("beta2", 0.999))),
    )

    sched_cfg = cfg.get("scheduler", {}) or {}
    sched_name = (sched_cfg.get("name", "none") or "none").strip().lower()
    scheduler = None
    if sched_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 2)),
            min_lr=float(sched_cfg.get("min_lr", 1.0e-6)),
        )
    elif sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get("t_max", cfg["epochs"])),
            eta_min=float(sched_cfg.get("eta_min", 1.0e-6)),
        )
    elif sched_name != "none":
        raise ValueError("scheduler.name must be one of: none, reduce_on_plateau, cosine")

    train_cfg = cfg.get("training", {}) or {}
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = -1.0
    es_cfg = cfg.get("early_stopping", {}) or {}
    patience = int(es_cfg.get("patience", 0))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    no_improve = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    ml_cfg = cfg.get("multilabel", {}) or {}
    threshold = float(ml_cfg.get("threshold", 0.5))

    for epoch in range(int(cfg["epochs"])):
        model.train()
        running = 0.0
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        model.eval()
        val_running_loss = 0.0
        tp = fp = fn = 0.0
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device).float()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels_batch)
                val_running_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs >= threshold).float()
                tp += (preds * labels_batch).sum().item()
                fp += (preds * (1.0 - labels_batch)).sum().item()
                fn += ((1.0 - preds) * labels_batch).sum().item()

        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        val_metric = (2.0 * precision * recall) / max(precision + recall, 1e-8)

        avg_train_loss = running / max(len(train_loader), 1)
        avg_val_loss = val_running_loss / max(len(val_loader), 1)

        if val_metric > best_val + min_delta:
            best_val = val_metric
            torch.save(model.state_dict(), output_dir / "best.pth")
            no_improve = 0
        else:
            no_improve += 1

        if scheduler is not None:
            if sched_name == "reduce_on_plateau":
                scheduler.step(val_metric)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch+1} train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} val_micro_f1={val_metric:.4f} lr={current_lr:.6g}"
        )

        if patience > 0 and no_improve >= patience:
            print(f"early_stop: no improvement for {patience} epochs")
            break

    torch.save(model.state_dict(), output_dir / "last.pth")

    val_logits = []
    val_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            logits = model(images).cpu()
            val_logits.append(logits)
            val_labels.append(labels_batch.float().cpu())
    if val_logits:
        logits = torch.cat(val_logits, dim=0)
        labels_t = torch.cat(val_labels, dim=0)
        preds = (torch.sigmoid(logits) >= threshold).float()
        tp_t, fp_t, fn_t = multilabel_counts_from_preds(preds, labels_t)
        metrics = multilabel_metrics_from_counts(tp_t, fp_t, fn_t)
        print(f"final_val_micro_f1={metrics['micro_f1']:.4f}")
        print(f"final_val_macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
