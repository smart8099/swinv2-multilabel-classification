# Synapse SwinV2 Multilabel Project

Dedicated project directory for Synapse multilabel classification using SwinV2 (`timm`).

## Setup
```bash
cd project_directory
python3 -m pip install -r requirements.txt
```

## Prepare Synapse data and splits (inside this project)
```bash
python3 scripts/prepare_synapse_classification.py \
  --synapse-root /path-to-root-folder-directory \
  --task Task002_Synapse \
  --out-root data/synapse_classification_all_organs_multilabel \
  --split-out outputs_synapse_all_organs_multilabel/splits \
  --split 0.8,0.2 \
  --case-split-mode train-val-only \
  --window 50,400 \
  --axis axial \
  --slice-step 1 \
  --class-scheme all-organs \
  --label-policy presence \
  --presence-min-pixels 30 \
  --output-format multilabel \
  --export-images-ts
```

## Train
```bash
python3 src/train_swinv2_multilabel.py --config configs/synapse_swinv2_multilabel.yaml
```

## Eval
```bash
python3 src/eval_swinv2_multilabel.py \
  --config configs/synapse_swinv2_multilabel.yaml \
  --weights outputs_synapse_all_organs_multilabel/best.pth \
  --tune-thresholds
```

## Export
```bash
python3 src/export_swinv2_multilabel.py \
  --config configs/synapse_swinv2_multilabel.yaml \
  --weights outputs_synapse_all_organs_multilabel/best.pth \
  --out outputs_synapse_all_organs_multilabel/swinv2_multilabel_export.pth
```

## Data contract
- Expects split JSON files under `output_dir/splits/`.
- Expects `labels.json` in `output_dir/`.
- Split item format: `{"path": "...", "labels": [0/1, ...]}`.
