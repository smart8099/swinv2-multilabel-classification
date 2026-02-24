from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from data import MultiLabelImageList


class SwinV2ClassificationWrapper(nn.Module):
    """Wrap SwinTransformerV2 backbone by pooling last stage feature map."""

    def __init__(self, backbone: nn.Module, num_classes: int, output_channels: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(output_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(self.pool(features[-1]).flatten(start_dim=1))


def _resolve_repo_root(model_cfg: dict) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    default_root = Path(__file__).resolve().parents[4] / "Swin-Transformer-V2"
    raw = model_cfg.get("repo_root", str(default_root))
    repo_root = Path(raw).expanduser()
    if not repo_root.is_absolute():
        repo_root = (project_root / repo_root).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"swin repo not found: {repo_root}")
    if not (repo_root / "swin_transformer_v2").exists():
        raise FileNotFoundError(f"missing swin_transformer_v2 package under: {repo_root}")
    return repo_root


def _import_swin_builders(repo_root: Path):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from swin_transformer_v2 import (
        swin_transformer_v2_t,
        swin_transformer_v2_s,
        swin_transformer_v2_b,
        swin_transformer_v2_l,
        swin_transformer_v2_h,
        swin_transformer_v2_g,
    )

    return {
        "t": swin_transformer_v2_t,
        "s": swin_transformer_v2_s,
        "b": swin_transformer_v2_b,
        "l": swin_transformer_v2_l,
        "h": swin_transformer_v2_h,
        "g": swin_transformer_v2_g,
    }


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_split(split_dir: Path):
    def _load(name: str):
        with open(split_dir / f"{name}.json", "r") as f:
            return json.load(f)

    return _load("train"), _load("val"), _load("test")


def compute_channel_stats(ds, batch_size: int, num_workers: int):
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    channel_sum = torch.zeros(3, dtype=torch.double)
    channel_sum_sq = torch.zeros(3, dtype=torch.double)
    num_pixels = 0
    for images, _ in loader:
        images = images.double()
        b, c, h, w = images.shape
        if c != 3:
            raise ValueError(f"expected 3 channels, got {c}")
        num_pixels += b * h * w
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images**2).sum(dim=[0, 2, 3])
    if num_pixels == 0:
        raise ValueError("no pixels found while computing normalization stats")
    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sum_sq / num_pixels - mean**2)
    return mean.tolist(), std.tolist()


def ensure_labels_file(output_dir: Path, labels):
    labels_path = output_dir / "labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump({"labels": labels}, f, indent=2)


def resolve_labels(output_dir: Path, train_items):
    labels_path = output_dir / "labels.json"
    if labels_path.exists():
        with open(labels_path, "r") as f:
            labels = json.load(f)["labels"]
        return labels

    if not train_items:
        raise ValueError("empty train split; cannot infer class count")

    first = train_items[0]
    if isinstance(first, dict):
        num_classes = len(first["labels"])
    else:
        num_classes = len(first[1])

    labels = [f"class_{i}" for i in range(num_classes)]
    ensure_labels_file(output_dir, labels)
    return labels


def build_transforms(cfg: dict, train_items):
    img_size = int(cfg["img_size"])
    aug = cfg.get("augmentation", {}) or {}
    color = aug.get("color_jitter", {}) or {}

    tf_train_parts = [transforms.Resize((img_size, img_size))]
    if aug.get("hflip", True):
        tf_train_parts.append(transforms.RandomHorizontalFlip())
    if aug.get("vflip", False):
        tf_train_parts.append(transforms.RandomVerticalFlip())
    if aug.get("rotation_deg", 0) > 0:
        tf_train_parts.append(transforms.RandomRotation(aug.get("rotation_deg", 0)))
    if color:
        tf_train_parts.append(
            transforms.ColorJitter(
                brightness=color.get("brightness", 0.0),
                contrast=color.get("contrast", 0.0),
                saturation=color.get("saturation", 0.0),
                hue=color.get("hue", 0.0),
            )
        )
    tf_train_parts.append(transforms.ToTensor())

    tf_eval_parts = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]

    norm_cfg = cfg.get("normalization", {}) or {}
    norm_enabled = bool(norm_cfg.get("enabled", False))
    stats_path = Path(norm_cfg.get("stats_path") or (Path(cfg["output_dir"]) / "normalize.json"))

    if norm_enabled:
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            mean = stats["mean"]
            std = stats["std"]
        else:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            tf_stats = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ]
            )
            ds_stats = MultiLabelImageList(train_items, transform=tf_stats)
            mean, std = compute_channel_stats(
                ds_stats,
                batch_size=int(cfg["batch_size"]),
                num_workers=int(cfg["num_workers"]),
            )
            with open(stats_path, "w") as f:
                json.dump({"mean": mean, "std": std}, f, indent=2)
            print(f"saved normalization stats: {stats_path}")

        tf_train_parts.append(transforms.Normalize(mean=mean, std=std))
        tf_eval_parts.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(tf_train_parts), transforms.Compose(tf_eval_parts)


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    model_cfg = cfg.get("model", {}) or {}
    repo_root = _resolve_repo_root(model_cfg)
    builders = _import_swin_builders(repo_root)

    variant = str(model_cfg.get("variant", "t")).strip().lower()
    if variant not in builders:
        raise ValueError("model.variant must be one of: t, s, b, l, h, g")
    print(f"swinv2_repo_root={repo_root}")
    print(f"swinv2_variant={variant}")

    output_channels = {
        "t": 768,
        "s": 768,
        "b": 1024,
        "l": 1536,
        "h": 2816,
        "g": 4096,
    }[variant]

    builder: Callable = builders[variant]
    img_size = int(cfg["img_size"])
    backbone = builder(
        input_resolution=(img_size, img_size),
        in_channels=3,
        window_size=int(model_cfg.get("window_size", 8)),
        use_checkpoint=bool(model_cfg.get("use_checkpoint", False)),
        sequential_self_attention=bool(model_cfg.get("sequential_self_attention", False)),
        use_deformable_block=bool(model_cfg.get("use_deformable_block", False)),
        dropout=float(model_cfg.get("drop_rate", 0.0)),
        dropout_attention=float(model_cfg.get("dropout_attention", 0.0)),
        dropout_path=float(model_cfg.get("drop_path_rate", 0.1)),
    )
    model = SwinV2ClassificationWrapper(
        backbone=backbone,
        num_classes=num_classes,
        output_channels=output_channels,
    )

    backbone_weights = str(model_cfg.get("backbone_weights", "") or "").strip()
    if backbone_weights:
        state = torch.load(Path(backbone_weights).expanduser(), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        backbone.load_state_dict(state, strict=True)

    return model


def collect_logits_labels(model: nn.Module, loader, device: torch.device):
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to(device)
            logits = model(images).cpu()
            logits_all.append(logits)
            labels_all.append(labels_batch.float().cpu())
    if not logits_all:
        return torch.empty((0, 0)), torch.empty((0, 0))
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def multilabel_counts_from_preds(preds: torch.Tensor, labels: torch.Tensor):
    tp = (preds * labels).sum(dim=0).double()
    fp = (preds * (1.0 - labels)).sum(dim=0).double()
    fn = ((1.0 - preds) * labels).sum(dim=0).double()
    return tp, fp, fn


def multilabel_metrics_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor):
    precision = tp / torch.clamp(tp + fp, min=1e-8)
    recall = tp / torch.clamp(tp + fn, min=1e-8)
    f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-8)

    tp_micro = tp.sum().item()
    fp_micro = fp.sum().item()
    fn_micro = fn.sum().item()
    p_micro = tp_micro / max(tp_micro + fp_micro, 1e-8)
    r_micro = tp_micro / max(tp_micro + fn_micro, 1e-8)
    f1_micro = (2.0 * p_micro * r_micro) / max(p_micro + r_micro, 1e-8)

    return {
        "micro_f1": f1_micro,
        "macro_f1": f1.mean().item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "per_class_f1": f1,
    }


def tune_per_class_thresholds(logits: torch.Tensor, labels: torch.Tensor):
    probs = torch.sigmoid(logits)
    num_classes = probs.size(1)
    candidates = torch.linspace(0.1, 0.9, steps=17)
    best_thresholds = torch.full((num_classes,), 0.5, dtype=torch.float32)

    for c in range(num_classes):
        y = labels[:, c]
        p = probs[:, c]
        best_f1 = -1.0
        best_t = 0.5
        for t in candidates:
            pred = (p >= t).float()
            tp = (pred * y).sum().item()
            fp = (pred * (1.0 - y)).sum().item()
            fn = ((1.0 - pred) * y).sum().item()
            precision = tp / max(tp + fp, 1e-8)
            recall = tp / max(tp + fn, 1e-8)
            f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t.item())
        best_thresholds[c] = best_t

    return best_thresholds


def tune_global_threshold_for_micro_f1(logits: torch.Tensor, labels: torch.Tensor):
    probs = torch.sigmoid(logits)
    candidates = torch.linspace(0.1, 0.9, steps=17)
    best_f1 = -1.0
    best_t = 0.5
    for t in candidates:
        preds = (probs >= t).float()
        tp, fp, fn = multilabel_counts_from_preds(preds, labels)
        f1_micro = multilabel_metrics_from_counts(tp, fp, fn)["micro_f1"]
        if f1_micro > best_f1:
            best_f1 = f1_micro
            best_t = float(t.item())
    return best_t
