from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


GROUPED_CLASS_MAP: Dict[str, List[int]] = {
    "Spleen": [1],
    "Kidney": [2, 3],
    "Gallbladder": [4],
    "Esophagus": [5],
    "Liver": [6],
    "Stomach": [7],
    "Vessels": [8, 9, 10],
    "Pancreas": [11],
    "Adrenal": [12, 13],
}

ALL_ORGANS_CLASS_MAP: Dict[str, List[int]] = {
    "Spleen": [1],
    "RightKidney": [2],
    "LeftKidney": [3],
    "Gallbladder": [4],
    "Esophagus": [5],
    "Liver": [6],
    "Stomach": [7],
    "Aorta": [8],
    "IVC": [9],
    "PortalSplenicVein": [10],
    "Pancreas": [11],
    "RightAdrenal": [12],
    "LeftAdrenal": [13],
}


def parse_split_values(split: str) -> List[float]:
    """Parse a comma-separated split string into raw float values.

    Accepts either "train,val" or "train,val,test". Validation of whether
    2-way or 3-way splits are allowed is handled later by split mode.
    """
    vals = [float(v.strip()) for v in split.split(",")]
    if len(vals) not in (2, 3):
        raise ValueError("--split must be train,val or train,val,test")
    if any(v < 0 for v in vals):
        raise ValueError("split values must be non-negative")
    total = sum(vals)
    if total <= 0:
        raise ValueError("split values must sum to > 0")
    return vals


def build_id_to_class(class_map: Dict[str, List[int]]) -> Dict[int, str]:
    """Invert class_map from class->organ_ids to organ_id->class.

    Raises if any organ ID is assigned to more than one class.
    """
    id_to_class: Dict[int, str] = {}
    for class_name, ids in class_map.items():
        for organ_id in ids:
            if organ_id in id_to_class:
                raise ValueError(f"organ id {organ_id} mapped more than once")
            id_to_class[organ_id] = class_name
    return id_to_class


def window_to_uint8(slice_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply fixed CT windowing and convert a 2D HU slice to uint8."""
    lo = center - (width / 2.0)
    hi = center + (width / 2.0)
    clipped = np.clip(slice_hu, lo, hi)
    scaled = (clipped - lo) / max(hi - lo, 1e-6)
    return (scaled * 255.0).astype(np.uint8)


def axis_to_int(axis: str) -> int:
    """Map plane name to volume axis index used for slicing."""
    lookup = {"axial": 2, "coronal": 1, "sagittal": 0}
    if axis not in lookup:
        raise ValueError(f"unsupported axis: {axis}")
    return lookup[axis]


def dominant_class_for_slice(mask2d: np.ndarray, id_to_class: Dict[int, str], min_pixels: int) -> str | None:
    """Return the largest-mask class in a slice if it passes min_pixels.

    Background (0) is ignored. Returns None when no mapped class exists or
    the dominant class area is below threshold.
    """
    labels, counts = np.unique(mask2d, return_counts=True)
    per_class: Counter[str] = Counter()
    for organ_id, count in zip(labels.tolist(), counts.tolist()):
        if organ_id == 0:
            continue
        class_name = id_to_class.get(int(organ_id))
        if class_name is None:
            continue
        per_class[class_name] += int(count)

    if not per_class:
        return None

    cls, pix = per_class.most_common(1)[0]
    if pix < min_pixels:
        return None
    return cls


def classes_present_for_slice(mask2d: np.ndarray, id_to_class: Dict[int, str], min_pixels: int) -> List[str]:
    """Return all classes present in a slice with area >= min_pixels.

    Background (0) and unknown IDs are ignored. Output is sorted for
    deterministic file naming and JSON generation.
    """
    labels, counts = np.unique(mask2d, return_counts=True)
    per_class: Counter[str] = Counter()
    for organ_id, count in zip(labels.tolist(), counts.tolist()):
        if organ_id == 0:
            continue
        class_name = id_to_class.get(int(organ_id))
        if class_name is None:
            continue
        per_class[class_name] += int(count)
    present = [k for k, v in per_class.items() if v >= min_pixels]
    present.sort()
    return present


def per_class_pixel_counts(mask2d: np.ndarray, id_to_class: Dict[int, str]) -> Counter[str]:
    """Count mapped foreground pixels per class in a slice."""
    labels, counts = np.unique(mask2d, return_counts=True)
    per_class: Counter[str] = Counter()
    for organ_id, count in zip(labels.tolist(), counts.tolist()):
        if organ_id == 0:
            continue
        class_name = id_to_class.get(int(organ_id))
        if class_name is None:
            continue
        per_class[class_name] += int(count)
    return per_class


def mask_to_color(mask2d: np.ndarray) -> np.ndarray:
    """Render integer label mask to an RGB array for quick visual debugging."""
    h, w = mask2d.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # Fixed palette (index by label id mod len)
    palette = np.array(
        [
            [0, 0, 0],
            [255, 99, 71],
            [60, 179, 113],
            [30, 144, 255],
            [255, 215, 0],
            [186, 85, 211],
            [0, 206, 209],
            [255, 140, 0],
            [100, 149, 237],
            [220, 20, 60],
            [154, 205, 50],
            [255, 105, 180],
            [0, 191, 255],
            [238, 130, 238],
        ],
        dtype=np.uint8,
    )
    labels = np.unique(mask2d)
    for lab in labels:
        if lab <= 0:
            continue
        color = palette[int(lab) % len(palette)]
        out[mask2d == lab] = color
    return out


def save_skipped_debug_panel(
    out_path: Path,
    img_u8: np.ndarray,
    mask2d: np.ndarray,
    case_id: str,
    axis: str,
    slice_idx: int,
    reason: str,
    threshold_name: str,
    threshold_value: int,
    foreground_pixels: int,
    per_class: Counter[str],
    overlay_alpha: float,
) -> None:
    """Save a side-by-side panel (image, mask, overlay) with text stats."""
    img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1).astype(np.uint8)
    mask_rgb = mask_to_color(mask2d)
    mask_present = (mask2d > 0)[..., None]
    overlay = img_rgb.copy().astype(np.float32)
    overlay[mask_present[:, :, 0]] = (
        (1.0 - overlay_alpha) * overlay[mask_present[:, :, 0]]
        + overlay_alpha * mask_rgb[mask_present[:, :, 0]]
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    h, w = img_u8.shape
    gap = 8
    footer_h = 110
    canvas = np.zeros((h + footer_h, w * 3 + gap * 4, 3), dtype=np.uint8)
    x0 = gap
    x1 = x0 + w + gap
    x2 = x1 + w + gap
    canvas[:h, x0 : x0 + w] = img_rgb
    canvas[:h, x1 : x1 + w] = mask_rgb
    canvas[:h, x2 : x2 + w] = overlay

    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)
    y = h + 8
    line1 = f"{case_id} {axis} idx={slice_idx} reason={reason}"
    line2 = f"foreground_pixels={foreground_pixels} {threshold_name}={threshold_value}"
    if per_class:
        pcs = ", ".join([f"{k}:{v}" for k, v in per_class.most_common(8)])
    else:
        pcs = "none"
    line3 = f"per_class_pixels: {pcs}"
    draw.text((8, y), line1, fill=(255, 255, 255))
    draw.text((8, y + 22), line2, fill=(255, 255, 255))
    draw.text((8, y + 44), line3, fill=(255, 255, 255))
    draw.text((x0, 4), "image", fill=(255, 255, 255))
    draw.text((x1, 4), "mask", fill=(255, 255, 255))
    draw.text((x2, 4), "overlay", fill=(255, 255, 255))
    img.save(out_path)


def load_task_root(synapse_root: Path, task_name: str) -> Path:
    """Resolve and validate the Synapse task root directory."""
    task_root = synapse_root / "unetr_pp_raw" / "unetr_pp_raw_data" / task_name
    if not task_root.exists():
        raise FileNotFoundError(f"Task root not found: {task_root}")
    for d in ["imagesTr", "labelsTr"]:
        if not (task_root / d).exists():
            raise FileNotFoundError(f"Missing {d} under {task_root}")
    return task_root


def load_dataset_meta(task_root: Path) -> dict:
    """Load task dataset metadata from dataset.json."""
    dataset_json = task_root / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"Missing dataset.json: {dataset_json}")
    with open(dataset_json, "r") as f:
        return json.load(f)


def case_id_from_path(path: Path) -> str:
    """Extract normalized case ID from image filename.

    Handles both nnU-Net style names (e.g., img0001_0000.nii.gz) and
    non-suffixed variants (e.g., img0001.nii.gz).
    """
    base = path.name
    if base.endswith(".nii.gz"):
        base = base[: -len(".nii.gz")]
    elif base.endswith(".nii"):
        base = base[: -len(".nii")]
    return base[:-5] if base.endswith("_0000") else base


def parse_dataset_training_list(task_root: Path, meta: dict) -> List[Tuple[Path, Path, str]]:
    """Build validated (image_path, label_path, case_id) training tuples.

    Includes fallback for datasets where dataset.json lists imgXXXX.nii.gz
    but files are stored as imgXXXX_0000.nii.gz.
    """

    pairs: List[Tuple[Path, Path, str]] = []
    for item in meta.get("training", []):
        image_rel = item["image"].replace("./", "")
        label_rel = item["label"].replace("./", "")
        image_path = task_root / image_rel
        label_path = task_root / label_rel

        # Some Synapse exports list imagesTr/imgXXXX.nii.gz in dataset.json
        # while files are stored as imgXXXX_0000.nii.gz (single modality suffix).
        if not image_path.exists() and image_path.name.endswith(".nii.gz"):
            alt_name = image_path.name.replace(".nii.gz", "_0000.nii.gz")
            alt_path = image_path.with_name(alt_name)
            if alt_path.exists():
                image_path = alt_path

        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing pair: {image_path} / {label_path}")

        case_id = case_id_from_path(image_path)
        pairs.append((image_path, label_path, case_id))

    if not pairs:
        raise ValueError("No training pairs found in dataset.json")

    return pairs


def parse_dataset_test_list(task_root: Path, meta: dict) -> List[Tuple[Path, str]]:
    """Build validated (image_path, case_id) list for unlabeled test cases."""
    items: List[Tuple[Path, str]] = []
    for image_rel_raw in meta.get("test", []):
        image_rel = image_rel_raw.replace("./", "")
        image_path = task_root / image_rel
        if not image_path.exists() and image_path.name.endswith(".nii.gz"):
            alt_name = image_path.name.replace(".nii.gz", "_0000.nii.gz")
            alt_path = image_path.with_name(alt_name)
            if alt_path.exists():
                image_path = alt_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing test image: {image_path}")
        items.append((image_path, case_id_from_path(image_path)))
    return items


def split_cases(
    case_ids: List[str], split_cfg: Dict[str, float], seed: int, split_mode: str
) -> Dict[str, set[str]]:
    """Split case IDs into train/val/(test) sets using deterministic shuffle.

    Splitting is case-level so all slices from the same case remain in the
    same partition, preventing patient leakage.
    """
    rng = random.Random(seed)
    ids = case_ids[:]
    rng.shuffle(ids)

    n = len(ids)
    if split_mode == "train-val-only":
        n_train = int(n * split_cfg["train"])
        train_ids = set(ids[:n_train])
        val_ids = set(ids[n_train:])
        test_ids: set[str] = set()
        if not train_ids or not val_ids:
            print("warning: one split has zero cases; consider changing --split")
        return {"train": train_ids, "val": val_ids, "test": test_ids}

    n_train = int(n * split_cfg["train"])
    n_val = int(n * split_cfg["val"])
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])
    if not train_ids or not val_ids or not test_ids:
        print("warning: one split has zero cases; consider changing --split")
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def choose_split(case_id: str, case_splits: Dict[str, set[str]]) -> str:
    """Return the split name for a given case ID."""
    for split_name, members in case_splits.items():
        if case_id in members:
            return split_name
    raise KeyError(f"case_id {case_id} missing from split map")


def main() -> None:
    """CLI entrypoint for Synapse-to-classification data preparation.

    The pipeline:
    1) Reads task metadata and resolves image/mask files.
    2) Splits cases at patient level.
    3) Slices 3D volumes into 2D images.
    4) Derives labels from masks (dominant or presence policy).
    5) Writes class folders, split JSONs, and summary metadata.
    6) Optionally exports unlabeled imagesTs slices for inference.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synapse-root",
        default="/Users/abdulbasit/Documents/phd_lifetime/endo_agent_project/DATASET_Synapse",
        help="Path to DATASET_Synapse root",
    )
    parser.add_argument("--task", default="Task002_Synapse", help="Task folder under .../unetr_pp_raw_data")
    parser.add_argument(
        "--out-root",
        default="data/synapse_classification",
        help="Output folder containing class subfolders",
    )
    parser.add_argument(
        "--split-out",
        default="outputs_synapse/splits",
        help="Output folder for train/val/test JSON files",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test")
    parser.add_argument(
        "--case-split-mode",
        default="train-val-test",
        choices=["train-val-test", "train-val-only"],
        help="Split labeled training cases into train/val/test or train/val only",
    )
    parser.add_argument(
        "--window",
        default="50,400",
        help="CT window center,width. Example: 50,400",
    )
    parser.add_argument(
        "--axis",
        default="axial",
        choices=["axial", "coronal", "sagittal"],
        help="Slice orientation for export",
    )
    parser.add_argument(
        "--min-mask-pixels",
        type=int,
        default=200,
        help="Skip slices whose dominant class has fewer than this many pixels",
    )
    parser.add_argument(
        "--slice-step",
        type=int,
        default=1,
        help="Keep every Nth slice",
    )
    parser.add_argument(
        "--max-slices-per-case",
        type=int,
        default=0,
        help="If > 0, cap exported slices per case after filtering (useful for quick dry-runs)",
    )
    parser.add_argument(
        "--class-scheme",
        default="grouped",
        choices=["grouped", "all-organs"],
        help="Label scheme: grouped(9 classes) or all-organs(13 classes from Synapse IDs)",
    )
    parser.add_argument(
        "--label-policy",
        default="dominant",
        choices=["dominant", "presence"],
        help="dominant: one class per slice; presence: one sample per class present in the slice",
    )
    parser.add_argument(
        "--presence-min-pixels",
        type=int,
        default=50,
        help="For label-policy=presence, require at least this many pixels per organ in a slice",
    )
    parser.add_argument(
        "--output-format",
        default="multiclass",
        choices=["multiclass", "multilabel"],
        help="multiclass writes [path,class_id] items; multilabel writes {'path','labels'} with multi-hot labels",
    )
    parser.add_argument(
        "--export-images-ts",
        action="store_true",
        help="Also export unlabeled imagesTs slices for inference",
    )
    parser.add_argument(
        "--images-ts-out",
        default="",
        help="Output dir for exported imagesTs slices (default: <out-root>/imagesTs_inference)",
    )
    parser.add_argument(
        "--images-ts-json",
        default="",
        help="JSON listing exported imagesTs slice paths (default: <split-out>/imagesTs_inference.json)",
    )
    parser.add_argument(
        "--save-skipped",
        action="store_true",
        help="Save skipped slices for visual inspection (image + mask preview + JSON manifest)",
    )
    parser.add_argument(
        "--skipped-out",
        default="",
        help="Output dir for skipped samples (default: <out-root>/skipped)",
    )
    parser.add_argument(
        "--save-skipped-debug",
        action="store_true",
        help="Save composite debug panels (image/mask/overlay + text stats) for skipped slices",
    )
    parser.add_argument(
        "--skipped-debug-out",
        default="",
        help="Output dir for skipped debug panels (default: <out-root>/skipped_debug)",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.35,
        help="Overlay alpha for skipped debug panels",
    )
    args = parser.parse_args()

    split_vals = parse_split_values(args.split)
    if args.case_split_mode == "train-val-only":
        if len(split_vals) == 2:
            train_val_sum = split_vals[0] + split_vals[1]
            split_cfg = {"train": split_vals[0] / train_val_sum, "val": split_vals[1] / train_val_sum}
        else:
            train_val_sum = split_vals[0] + split_vals[1]
            split_cfg = {"train": split_vals[0] / train_val_sum, "val": split_vals[1] / train_val_sum}
    else:
        if len(split_vals) != 3:
            raise ValueError("--split must be train,val,test when --case-split-mode=train-val-test")
        total = split_vals[0] + split_vals[1] + split_vals[2]
        split_cfg = {
            "train": split_vals[0] / total,
            "val": split_vals[1] / total,
            "test": split_vals[2] / total,
        }
    class_map = ALL_ORGANS_CLASS_MAP if args.class_scheme == "all-organs" else GROUPED_CLASS_MAP
    id_to_class = build_id_to_class(class_map)

    center_str, width_str = args.window.split(",")
    window_center = float(center_str)
    window_width = float(width_str)

    if args.slice_step < 1:
        raise ValueError("--slice-step must be >= 1")

    synapse_root = Path(args.synapse_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    split_out = Path(args.split_out).expanduser().resolve()

    task_root = load_task_root(synapse_root, args.task)
    meta = load_dataset_meta(task_root)
    pairs = parse_dataset_training_list(task_root, meta)

    case_ids = [case_id for _, _, case_id in pairs]
    case_splits = split_cases(case_ids, split_cfg, args.seed, args.case_split_mode)

    labels = sorted(class_map.keys())
    label_to_id = {name: i for i, name in enumerate(labels)}
    if args.output_format == "multiclass":
        for name in labels:
            (out_root / name).mkdir(parents=True, exist_ok=True)
    else:
        (out_root / "images").mkdir(parents=True, exist_ok=True)
    split_out.mkdir(parents=True, exist_ok=True)
    with open(split_out.parent / "labels.json", "w") as f:
        json.dump({"labels": labels}, f, indent=2)

    split_items: Dict[str, List] = {"train": [], "val": [], "test": []}
    class_counter = Counter()
    skipped_slices = 0
    skipped_records: List[dict] = []

    skipped_out_dir = None
    if args.save_skipped:
        skipped_out_dir = (
            Path(args.skipped_out).expanduser().resolve()
            if args.skipped_out
            else out_root / "skipped"
        )
        skipped_out_dir.mkdir(parents=True, exist_ok=True)
    skipped_debug_out_dir = None
    if args.save_skipped_debug:
        skipped_debug_out_dir = (
            Path(args.skipped_debug_out).expanduser().resolve()
            if args.skipped_debug_out
            else out_root / "skipped_debug"
        )
        skipped_debug_out_dir.mkdir(parents=True, exist_ok=True)

    axis = axis_to_int(args.axis)

    for image_path, label_path, case_id in pairs:
        vol = nib.load(str(image_path)).get_fdata(dtype=np.float32)
        seg = nib.load(str(label_path)).get_fdata(dtype=np.float32).astype(np.int16)

        if vol.shape != seg.shape:
            raise ValueError(f"shape mismatch for {case_id}: {vol.shape} vs {seg.shape}")

        vol = np.asarray(vol)
        seg = np.asarray(seg)

        num_slices = vol.shape[axis]
        saved_for_case = 0

        for idx in range(0, num_slices, args.slice_step):
            if axis == 2:
                img2d = vol[:, :, idx]
                mask2d = seg[:, :, idx]
            elif axis == 1:
                img2d = vol[:, idx, :]
                mask2d = seg[:, idx, :]
            else:
                img2d = vol[idx, :, :]
                mask2d = seg[idx, :, :]

            img_u8 = window_to_uint8(img2d, center=window_center, width=window_width)
            split_name = choose_split(case_id, case_splits)

            if args.label_policy == "dominant":
                class_name = dominant_class_for_slice(mask2d, id_to_class, args.min_mask_pixels)
                if class_name is None:
                    skipped_slices += 1
                    per_class = per_class_pixel_counts(mask2d, id_to_class)
                    max_pixels = max(per_class.values()) if per_class else 0
                    reason = "no_mapped_foreground" if not per_class else "below_min_mask_pixels"
                    if args.save_skipped and skipped_out_dir is not None:
                        base = f"{case_id}_{args.axis}_{idx:04d}"
                        img_path = skipped_out_dir / f"{base}_img.png"
                        mask_path = skipped_out_dir / f"{base}_mask.png"
                        Image.fromarray(img_u8, mode="L").save(img_path)
                        mask_u8 = np.clip(mask2d.astype(np.int16), 0, 255).astype(np.uint8)
                        Image.fromarray(mask_u8, mode="L").save(mask_path)
                        skipped_records.append(
                            {
                                "case_id": case_id,
                                "axis": args.axis,
                                "slice_index": idx,
                                "reason": reason,
                                "max_class_pixels": int(max_pixels),
                                "min_mask_pixels": int(args.min_mask_pixels),
                                "foreground_pixels": int((mask2d > 0).sum()),
                                "image_path": str(img_path),
                                "mask_path": str(mask_path),
                            }
                        )
                    if args.save_skipped_debug and skipped_debug_out_dir is not None:
                        per_class = per_class_pixel_counts(mask2d, id_to_class)
                        base = f"{case_id}_{args.axis}_{idx:04d}"
                        dbg_path = skipped_debug_out_dir / f"{base}_debug.png"
                        save_skipped_debug_panel(
                            out_path=dbg_path,
                            img_u8=img_u8,
                            mask2d=mask2d,
                            case_id=case_id,
                            axis=args.axis,
                            slice_idx=idx,
                            reason=reason,
                            threshold_name="min_mask_pixels",
                            threshold_value=int(args.min_mask_pixels),
                            foreground_pixels=int((mask2d > 0).sum()),
                            per_class=per_class,
                            overlay_alpha=float(args.overlay_alpha),
                        )
                    continue
                if args.output_format == "multiclass":
                    out_path = out_root / class_name / f"{case_id}_{args.axis}_{idx:04d}.png"
                    target = label_to_id[class_name]
                else:
                    out_path = out_root / "images" / f"{case_id}_{args.axis}_{idx:04d}.png"
                    one_hot = [0] * len(labels)
                    one_hot[label_to_id[class_name]] = 1
                    target = one_hot
                Image.fromarray(img_u8, mode="L").save(out_path)
                if args.output_format == "multiclass":
                    split_items[split_name].append((str(out_path), target))
                else:
                    split_items[split_name].append({"path": str(out_path), "labels": target})
                class_counter[class_name] += 1
                saved_for_case += 1
            else:
                present_classes = classes_present_for_slice(
                    mask2d, id_to_class, min_pixels=args.presence_min_pixels
                )
                if not present_classes:
                    skipped_slices += 1
                    per_class = per_class_pixel_counts(mask2d, id_to_class)
                    max_pixels = max(per_class.values()) if per_class else 0
                    reason = "no_mapped_foreground" if not per_class else "below_presence_min_pixels"
                    if args.save_skipped and skipped_out_dir is not None:
                        base = f"{case_id}_{args.axis}_{idx:04d}"
                        img_path = skipped_out_dir / f"{base}_img.png"
                        mask_path = skipped_out_dir / f"{base}_mask.png"
                        Image.fromarray(img_u8, mode="L").save(img_path)
                        mask_u8 = np.clip(mask2d.astype(np.int16), 0, 255).astype(np.uint8)
                        Image.fromarray(mask_u8, mode="L").save(mask_path)
                        skipped_records.append(
                            {
                                "case_id": case_id,
                                "axis": args.axis,
                                "slice_index": idx,
                                "reason": reason,
                                "max_class_pixels": int(max_pixels),
                                "presence_min_pixels": int(args.presence_min_pixels),
                                "foreground_pixels": int((mask2d > 0).sum()),
                                "image_path": str(img_path),
                                "mask_path": str(mask_path),
                            }
                        )
                    if args.save_skipped_debug and skipped_debug_out_dir is not None:
                        per_class = per_class_pixel_counts(mask2d, id_to_class)
                        base = f"{case_id}_{args.axis}_{idx:04d}"
                        dbg_path = skipped_debug_out_dir / f"{base}_debug.png"
                        save_skipped_debug_panel(
                            out_path=dbg_path,
                            img_u8=img_u8,
                            mask2d=mask2d,
                            case_id=case_id,
                            axis=args.axis,
                            slice_idx=idx,
                            reason=reason,
                            threshold_name="presence_min_pixels",
                            threshold_value=int(args.presence_min_pixels),
                            foreground_pixels=int((mask2d > 0).sum()),
                            per_class=per_class,
                            overlay_alpha=float(args.overlay_alpha),
                        )
                    continue
                if args.output_format == "multiclass":
                    for class_name in present_classes:
                        out_path = out_root / class_name / f"{case_id}_{args.axis}_{idx:04d}_{class_name}.png"
                        Image.fromarray(img_u8, mode="L").save(out_path)
                        split_items[split_name].append((str(out_path), label_to_id[class_name]))
                        class_counter[class_name] += 1
                        saved_for_case += 1
                else:
                    out_path = out_root / "images" / f"{case_id}_{args.axis}_{idx:04d}.png"
                    Image.fromarray(img_u8, mode="L").save(out_path)
                    one_hot = [0] * len(labels)
                    for class_name in present_classes:
                        one_hot[label_to_id[class_name]] = 1
                        class_counter[class_name] += 1
                    split_items[split_name].append({"path": str(out_path), "labels": one_hot})
                    saved_for_case += 1

            if args.max_slices_per_case > 0 and saved_for_case >= args.max_slices_per_case:
                break

    ts_export_count = 0
    if args.export_images_ts:
        ts_items = parse_dataset_test_list(task_root, meta)
        images_ts_out = (
            Path(args.images_ts_out).expanduser().resolve()
            if args.images_ts_out
            else out_root / "imagesTs_inference"
        )
        images_ts_out.mkdir(parents=True, exist_ok=True)
        ts_json_path = (
            Path(args.images_ts_json).expanduser().resolve()
            if args.images_ts_json
            else split_out / "imagesTs_inference.json"
        )
        ts_payload: List[Tuple[str, str]] = []
        for image_path, case_id in ts_items:
            vol = nib.load(str(image_path)).get_fdata(dtype=np.float32)
            vol = np.asarray(vol)
            num_slices = vol.shape[axis]
            saved_for_case = 0
            case_out = images_ts_out / case_id
            case_out.mkdir(parents=True, exist_ok=True)
            for idx in range(0, num_slices, args.slice_step):
                if axis == 2:
                    img2d = vol[:, :, idx]
                elif axis == 1:
                    img2d = vol[:, idx, :]
                else:
                    img2d = vol[idx, :, :]
                img_u8 = window_to_uint8(img2d, center=window_center, width=window_width)
                out_path = case_out / f"{case_id}_{args.axis}_{idx:04d}.png"
                Image.fromarray(img_u8, mode="L").save(out_path)
                ts_payload.append((str(out_path), case_id))
                ts_export_count += 1
                saved_for_case += 1
                if args.max_slices_per_case > 0 and saved_for_case >= args.max_slices_per_case:
                    break
        with open(ts_json_path, "w") as f:
            json.dump(ts_payload, f)

    for split_name in ["train", "val", "test"]:
        with open(split_out / f"{split_name}.json", "w") as f:
            json.dump(split_items[split_name], f)
    if args.save_skipped and skipped_out_dir is not None:
        with open(split_out / "skipped_manifest.json", "w") as f:
            json.dump(skipped_records, f, indent=2)

    summary = {
        "task_root": str(task_root),
        "out_root": str(out_root),
        "split_out": str(split_out),
        "labels": labels,
        "class_counts": dict(class_counter),
        "split_counts": {k: len(v) for k, v in split_items.items()},
        "skipped_slices": skipped_slices,
        "num_cases": len(pairs),
        "axis": args.axis,
        "window": {"center": window_center, "width": window_width},
        "slice_step": args.slice_step,
        "min_mask_pixels": args.min_mask_pixels,
        "class_scheme": args.class_scheme,
        "label_policy": args.label_policy,
        "presence_min_pixels": args.presence_min_pixels,
        "output_format": args.output_format,
        "case_split_mode": args.case_split_mode,
        "effective_split": split_cfg,
        "images_ts_exported": bool(args.export_images_ts),
        "images_ts_slice_count": ts_export_count,
        "save_skipped": bool(args.save_skipped),
        "skipped_out": str(skipped_out_dir) if skipped_out_dir is not None else None,
        "save_skipped_debug": bool(args.save_skipped_debug),
        "skipped_debug_out": str(skipped_debug_out_dir) if skipped_debug_out_dir is not None else None,
        "overlay_alpha": float(args.overlay_alpha),
    }
    with open(split_out / "synapse_prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("prepared Synapse classification dataset")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
