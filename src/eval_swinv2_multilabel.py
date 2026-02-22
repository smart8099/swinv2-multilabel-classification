from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import MultiLabelImageList
from swinv2_multilabel import (
    build_model,
    build_transforms,
    collect_logits_labels,
    load_config,
    load_split,
    multilabel_counts_from_preds,
    multilabel_metrics_from_counts,
    tune_global_threshold_for_micro_f1,
    tune_per_class_thresholds,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synapse_swinv2_multilabel.yaml")
    parser.add_argument("--weights", default="outputs_synapse_all_organs_multilabel/best.pth")
    parser.add_argument(
        "--tune-thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tune thresholds on val split before test evaluation.",
    )
    parser.add_argument(
        "--tune-mode",
        choices=["global_micro", "per_class"],
        default="global_micro",
        help="Threshold tuning mode when --tune-thresholds is enabled.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    task_type = (cfg.get("task_type", "multilabel") or "multilabel").strip().lower()
    if task_type != "multilabel":
        raise ValueError("eval_swinv2_multilabel.py only supports task_type=multilabel")

    split_dir = Path(cfg["output_dir"]) / "splits"
    train_items, val_items, test_items = load_split(split_dir)
    has_test_split = len(test_items) > 0
    if not has_test_split:
        print("warning: test split is empty; evaluating on val split instead")
        test_items = val_items

    labels_path = Path(cfg["output_dir"]) / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"missing labels file: {labels_path}")
    with open(labels_path, "r") as f:
        labels = json.load(f)["labels"]

    _, tf_eval = build_transforms(cfg, train_items)

    val_ds = MultiLabelImageList(val_items, transform=tf_eval)
    test_ds = MultiLabelImageList(test_items, transform=tf_eval)

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_classes=len(labels))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    ml_cfg = cfg.get("multilabel", {}) or {}
    default_threshold = float(ml_cfg.get("threshold", 0.5))
    thresholds = torch.full((len(labels),), default_threshold, dtype=torch.float32)

    if args.tune_thresholds:
        val_logits, val_labels = collect_logits_labels(model, val_loader, device)
        if val_logits.numel() > 0 and val_logits.shape[1] == len(labels):
            if args.tune_mode == "per_class":
                thresholds = tune_per_class_thresholds(val_logits, val_labels)
                print("thresholds_tuned_on=val mode=per_class")
                for i, name in enumerate(labels):
                    print(f"  {name}: {thresholds[i].item():.2f}")
            else:
                best_t = tune_global_threshold_for_micro_f1(val_logits, val_labels)
                thresholds = torch.full((len(labels),), best_t, dtype=torch.float32)
                print(f"thresholds_tuned_on=val mode=global_micro threshold={best_t:.2f}")
            if not has_test_split:
                print("warning: tuned and evaluated on val (test split is empty)")
        else:
            print("warning: could not tune thresholds from val split; using config threshold")

    test_logits, test_labels = collect_logits_labels(model, test_loader, device)
    test_probs = torch.sigmoid(test_logits)
    preds = (test_probs >= thresholds.unsqueeze(0)).float()
    tp, fp, fn = multilabel_counts_from_preds(preds, test_labels)
    metrics = multilabel_metrics_from_counts(tp, fp, fn)

    print(f"test_micro_f1={metrics['micro_f1']:.4f}")
    print(f"test_macro_f1={metrics['macro_f1']:.4f}")
    print(f"test_macro_precision={metrics['macro_precision']:.4f}")
    print(f"test_macro_recall={metrics['macro_recall']:.4f}")
    print("per_class_f1:")
    for i, name in enumerate(labels):
        print(f"  {name}: {metrics['per_class_f1'][i].item():.4f}")


if __name__ == "__main__":
    main()
