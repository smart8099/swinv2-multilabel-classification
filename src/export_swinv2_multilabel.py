from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from swinv2_multilabel import build_model, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synapse_swinv2_multilabel.yaml")
    parser.add_argument("--weights", default="outputs_synapse_all_organs_multilabel/best.pth")
    parser.add_argument("--out", default="outputs_synapse_all_organs_multilabel/swinv2_multilabel_export.pth")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    labels_path = Path(cfg["output_dir"]) / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"missing labels file: {labels_path}")
    with open(labels_path, "r") as f:
        labels = json.load(f)["labels"]

    model = build_model(cfg, num_classes=len(labels))
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "labels": labels,
        "model_name": (cfg.get("model", {}) or {}).get("name", "swinv2_tiny_window8_256"),
        "img_size": int(cfg["img_size"]),
        "task_type": "multilabel",
    }
    torch.save(payload, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
