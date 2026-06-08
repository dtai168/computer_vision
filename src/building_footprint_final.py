from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[no-redef]
        return iterable if iterable is not None else []


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
METRIC_COLUMNS = [
    "iou",
    "dice",
    "precision",
    "recall",
    "f1",
    "n_pred",
    "n_gt",
    "count_err",
    "area_ratio_pred",
    "area_ratio_gt",
    "area_abs_error",
]


@dataclass
class ExperimentConfig:
    random_seed: int = 42
    patch_size: int = 512
    patch_stride: int = 512
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    max_train_patches: int | None = 1200
    max_val_patches: int | None = 300
    max_test_patches: int | None = 300

    kmeans_clusters: int = 4
    kmeans_iter: int = 10
    kmeans_eps: float = 1.0

    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    otsu_clahe_clip_limit: float = 1.5
    morph_close_ksize: int = 9
    morph_open_ksize: int = 5
    min_building_area: int = 200

    pixels_per_train_patch: int = 1200
    max_train_pixels: int = 200_000
    svm_c_grid: tuple[float, ...] = (0.1, 0.5, 1.0)
    svm_max_iter: int = 5000
    max_svm_val_patches_for_tuning: int | None = 120

    unet_epochs: int = 20
    unet_batch_size: int = 4
    unet_learning_rate: float = 1e-3
    unet_weight_decay: float = 1e-4
    unet_patience: int = 5
    unet_num_workers: int = 2
    unet_use_cuda: bool = True
    unet_threshold_grid: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    max_unet_val_patches_for_threshold: int | None = 120
    unet_apply_morphology: bool = True

    qualitative_examples: int = 8
    output_dir: str = "outputs"
    dataset_root: str | None = None
    candidate_base_dirs: tuple[str, ...] = (
        "/kaggle/input/datasets/sagar100rathod/inria-aerial-image-labeling-dataset/AerialImageDataset",
        "/kaggle/input/inria-aerial-image-labeling-dataset/AerialImageDataset",
        "/kaggle/input/inria-aerial-image-labeling/AerialImageDataset",
        "/kaggle/input/aerial-image-dataset/AerialImageDataset",
        "D:/datasets/AerialImageDataset",
        "G:/AerialImageDataset",
    )

    def output_path(self) -> Path:
        return Path(self.output_dir)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_output_dirs(cfg: ExperimentConfig) -> dict[str, Path]:
    root = cfg.output_path()
    paths = {
        "root": root,
        "metrics": root / "metrics",
        "figures": root / "figures",
        "models": root / "models",
        "reports": root / "reports",
        "manifests": root / "manifests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_config(cfg: ExperimentConfig) -> None:
    paths = ensure_output_dirs(cfg)
    with (paths["root"] / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=_json_default)


def resolve_dataset_root(cfg: ExperimentConfig) -> Path:
    candidates: list[Path] = []
    if cfg.dataset_root:
        candidates.append(Path(cfg.dataset_root))
    if os.getenv("INRIA_DATASET_ROOT"):
        candidates.append(Path(os.environ["INRIA_DATASET_ROOT"]))
    candidates.extend(Path(p) for p in cfg.candidate_base_dirs)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for train_dir in kaggle_input.rglob("train"):
            if not train_dir.is_dir():
                continue
            root = train_dir.parent
            if (root / "train" / "images").exists() and (root / "train" / "gt").exists():
                candidates.append(root)

    for root in candidates:
        if (root / "train" / "images").exists() and (root / "train" / "gt").exists():
            return root

    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Inria dataset root was not found. Set cfg.dataset_root or INRIA_DATASET_ROOT.\n"
        f"Checked:\n{checked}"
    )


def get_dataset_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "images": root / "train" / "images",
        "masks": root / "train" / "gt",
    }


def city_from_name(stem: str) -> str:
    city = "".join(ch for ch in stem if not ch.isdigit())
    return city.strip("_-") or "unknown"


def mask_path_for_image(image_path: Path, mask_dir: Path) -> Path | None:
    same_suffix = mask_dir / image_path.name
    if same_suffix.exists():
        return same_suffix
    for suffix in IMAGE_SUFFIXES:
        candidate = mask_dir / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def discover_image_names(images_dir: Path, mask_dir: Path) -> list[str]:
    names: list[str] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if mask_path_for_image(image_path, mask_dir) is not None:
            names.append(image_path.name)
    if not names:
        raise RuntimeError(f"No image/mask pairs found in {images_dir} and {mask_dir}")
    return names


def split_image_names(names: list[str], cfg: ExperimentConfig) -> dict[str, list[str]]:
    if not math.isclose(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    shuffled = sorted(names)
    rng = random.Random(cfg.random_seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(n_total * cfg.train_ratio))
    n_val = int(round(n_total * cfg.val_ratio))
    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def validate_disjoint_splits(splits: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for split, names in splits.items():
        for name in names:
            if name in seen:
                raise AssertionError(f"Image {name} appears in both {seen[name]} and {split}")
            seen[name] = split


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask


def tile_positions(height: int, width: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    if height < patch_size or width < patch_size:
        return []
    positions: list[tuple[int, int]] = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            positions.append((y, x))
    return positions


def build_patch_records(
    split_name: str,
    image_names: list[str],
    dataset_paths: dict[str, Path],
    cfg: ExperimentConfig,
    max_patches: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for image_name in image_names:
        image_path = dataset_paths["images"] / image_name
        mask_path = mask_path_for_image(image_path, dataset_paths["masks"])
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for {image_path.name}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Could not read pair: {image_path}, {mask_path}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image and mask shape mismatch: {image_path.name}")

        height, width = image.shape[:2]
        for y, x in tile_positions(height, width, cfg.patch_size, cfg.patch_stride):
            records.append(
                {
                    "patch_id": f"{split_name}_{Path(image_name).stem}_{y}_{x}",
                    "split": split_name,
                    "image_name": image_name,
                    "city": city_from_name(Path(image_name).stem),
                    "x": x,
                    "y": y,
                    "patch_size": cfg.patch_size,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
            )

    if max_patches is not None and len(records) > max_patches:
        rng = random.Random(cfg.random_seed + {"train": 11, "val": 23, "test": 37}[split_name])
        records = rng.sample(records, max_patches)
        records = sorted(records, key=lambda r: (r["image_name"], r["y"], r["x"]))
    return records


def prepare_manifests(cfg: ExperimentConfig) -> tuple[dict[str, Path], dict[str, list[dict[str, Any]]]]:
    set_global_seed(cfg.random_seed)
    output_paths = ensure_output_dirs(cfg)
    dataset_root = resolve_dataset_root(cfg)
    dataset_paths = get_dataset_paths(dataset_root)

    image_names = discover_image_names(dataset_paths["images"], dataset_paths["masks"])
    splits = split_image_names(image_names, cfg)
    validate_disjoint_splits(splits)

    image_rows = []
    for split, names in splits.items():
        for name in names:
            image_rows.append({"split": split, "image_name": name, "city": city_from_name(Path(name).stem)})
    pd.DataFrame(image_rows).to_csv(output_paths["manifests"] / "image_split_manifest.csv", index=False)

    max_by_split = {
        "train": cfg.max_train_patches,
        "val": cfg.max_val_patches,
        "test": cfg.max_test_patches,
    }
    records_by_split = {
        split: build_patch_records(split, names, dataset_paths, cfg, max_by_split[split])
        for split, names in splits.items()
    }
    all_records = [record for records in records_by_split.values() for record in records]
    pd.DataFrame(all_records).to_csv(output_paths["manifests"] / "split_manifest.csv", index=False)
    for split, records in records_by_split.items():
        pd.DataFrame(records).to_csv(output_paths["manifests"] / f"{split}_manifest.csv", index=False)

    save_config(cfg)
    return output_paths, records_by_split


def read_patch(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    image = load_rgb(Path(record["image_path"]))
    mask = load_mask(Path(record["mask_path"]))
    x = int(record["x"])
    y = int(record["y"])
    patch_size = int(record["patch_size"])
    patch = image[y : y + patch_size, x : x + patch_size]
    gt = mask[y : y + patch_size, x : x + patch_size]
    return patch, gt


def binary_mask(mask: np.ndarray) -> np.ndarray:
    return mask > 127


def preprocess_patch(patch: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def segment_kmeans(patch: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).astype(np.float32)
    rgb = patch.astype(np.float32)
    combined = np.concatenate([rgb / 255.0, hsv / 255.0], axis=2)
    height, width, channels = combined.shape
    pixels = combined.reshape(-1, channels).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        cfg.kmeans_iter,
        cfg.kmeans_eps,
    )
    _, labels, centers = cv2.kmeans(
        pixels,
        cfg.kmeans_clusters,
        None,
        criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(height, width)
    centers = np.asarray(centers)
    building_label = int(np.argmax(centers[:, 5]))
    return np.where(labels == building_label, 255, 0).astype(np.uint8)


def segment_otsu(patch: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=cfg.otsu_clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid)
    saturation = clahe.apply(saturation)
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def morphological_cleanup(mask: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_close_ksize, cfg.morph_close_ksize))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_open_ksize, cfg.morph_open_ksize))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    output = np.zeros_like(cleaned)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= cfg.min_building_area:
            output[labels == label] = 255
    return output


def count_buildings(mask: np.ndarray) -> int:
    n_labels, _ = cv2.connectedComponents((mask > 127).astype(np.uint8), connectivity=8)
    return int(max(0, n_labels - 1))


def evaluate_single(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_b = binary_mask(pred)
    gt_b = binary_mask(gt)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    pred_sum = int(pred_b.sum())
    gt_sum = int(gt_b.sum())
    union = int(np.logical_or(pred_b, gt_b).sum())

    iou = 1.0 if union == 0 else float(tp / union)
    dice = 1.0 if pred_sum + gt_sum == 0 else float(2 * tp / (pred_sum + gt_sum))
    precision = 1.0 if pred_sum == 0 and gt_sum == 0 else (float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0)
    recall = 1.0 if pred_sum == 0 and gt_sum == 0 else (float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    n_pred = count_buildings(pred)
    n_gt = count_buildings(gt)
    area_ratio_pred = float(pred_sum / pred_b.size * 100.0)
    area_ratio_gt = float(gt_sum / gt_b.size * 100.0)
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": float(n_pred),
        "n_gt": float(n_gt),
        "count_err": float(abs(n_pred - n_gt)),
        "area_ratio_pred": area_ratio_pred,
        "area_ratio_gt": area_ratio_gt,
        "area_abs_error": float(abs(area_ratio_pred - area_ratio_gt)),
    }


def aggregate_summary(per_patch_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in per_patch_df.groupby("method", sort=False):
        row: dict[str, Any] = {
            "method": method,
            "n_patches": int(len(group)),
            "seconds_total": float(group["seconds"].sum()),
            "seconds_per_patch": float(group["seconds"].mean()),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_method(
    method_name: str,
    records: list[dict[str, Any]],
    predictor: Callable[[np.ndarray], np.ndarray],
    cfg: ExperimentConfig,
    show_progress: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    iterator = tqdm(records, desc=f"Evaluate {method_name}", disable=not show_progress)
    for record in iterator:
        patch, gt = read_patch(record)
        start = time.perf_counter()
        pred = predictor(patch)
        seconds = time.perf_counter() - start
        metrics = evaluate_single(pred, gt)
        rows.append(
            {
                "method": method_name,
                "patch_id": record["patch_id"],
                "split": record["split"],
                "image_name": record["image_name"],
                "city": record["city"],
                "x": int(record["x"]),
                "y": int(record["y"]),
                "seconds": seconds,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def extract_pixel_features(patch: np.ndarray) -> np.ndarray:
    rgb = patch.astype(np.float32) / 255.0
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0

    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    grad_mag = np.clip(grad_mag / 4.0, 0.0, 1.0)

    local_mean = cv2.blur(gray, (9, 9))
    local_sq_mean = cv2.blur(gray * gray, (9, 9))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))

    features = np.dstack(
        [
            rgb,
            lab,
            hsv,
            gray[:, :, None],
            grad_mag[:, :, None],
            local_mean[:, :, None],
            local_std[:, :, None],
        ]
    )
    return features.reshape(-1, features.shape[-1]).astype(np.float32)


def sample_pixels_from_patch(
    features: np.ndarray,
    mask: np.ndarray,
    pixels_per_patch: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    labels = (mask.reshape(-1) > 127).astype(np.uint8)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)

    if len(pos_idx) > 0 and len(neg_idx) > 0:
        n_pos = min(len(pos_idx), pixels_per_patch // 2)
        n_neg = min(len(neg_idx), pixels_per_patch - n_pos)
        chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        chosen = np.concatenate([chosen_pos, chosen_neg])
    elif len(pos_idx) > 0:
        chosen = rng.choice(pos_idx, size=min(len(pos_idx), pixels_per_patch), replace=False)
    else:
        n_neg = min(len(neg_idx), max(200, pixels_per_patch // 3))
        chosen = rng.choice(neg_idx, size=n_neg, replace=False)

    rng.shuffle(chosen)
    return features[chosen], labels[chosen]


def build_training_matrix(records: list[dict[str, Any]], cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for record in tqdm(records, desc="Extract SVM train features"):
        patch, gt = read_patch(record)
        processed = preprocess_patch(patch, cfg)
        features = extract_pixel_features(processed)
        x_sample, y_sample = sample_pixels_from_patch(features, gt, cfg.pixels_per_train_patch, rng)
        x_parts.append(x_sample)
        y_parts.append(y_sample)

    x_train = np.vstack(x_parts).astype(np.float32)
    y_train = np.concatenate(y_parts).astype(np.uint8)
    if len(y_train) > cfg.max_train_pixels:
        keep = rng.choice(np.arange(len(y_train)), size=cfg.max_train_pixels, replace=False)
        x_train = x_train[keep]
        y_train = y_train[keep]
    return x_train, y_train


def train_svm(records: list[dict[str, Any]], cfg: ExperimentConfig, c_value: float) -> tuple[Any, float]:
    x_train, y_train = build_training_matrix(records, cfg)
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual=False,
            max_iter=cfg.svm_max_iter,
            random_state=cfg.random_seed,
        ),
    )
    start = time.perf_counter()
    model.fit(x_train, y_train)
    return model, time.perf_counter() - start


def predict_svm_patch(patch: np.ndarray, model: Any, cfg: ExperimentConfig) -> np.ndarray:
    processed = preprocess_patch(patch, cfg)
    features = extract_pixel_features(processed)
    labels = model.predict(features).reshape(patch.shape[:2])
    raw = (labels.astype(np.uint8) * 255)
    return morphological_cleanup(raw, cfg)


def tune_svm(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    cfg: ExperimentConfig,
) -> tuple[Any, pd.DataFrame, float]:
    tuning_records = val_records
    if cfg.max_svm_val_patches_for_tuning is not None:
        tuning_records = val_records[: cfg.max_svm_val_patches_for_tuning]

    rows: list[dict[str, Any]] = []
    best_model: Any = None
    best_c = cfg.svm_c_grid[0]
    best_dice = -1.0
    for c_value in cfg.svm_c_grid:
        model, train_seconds = train_svm(train_records, cfg, c_value)
        val_df = evaluate_method(
            f"SVM_C_{c_value}",
            tuning_records,
            lambda patch, m=model: predict_svm_patch(patch, m, cfg),
            cfg,
        )
        dice = float(val_df["dice"].mean())
        rows.append({"C": c_value, "val_dice": dice, "train_seconds": train_seconds})
        if dice > best_dice:
            best_dice = dice
            best_model = model
            best_c = c_value

    output_paths = ensure_output_dirs(cfg)
    joblib.dump(best_model, output_paths["models"] / "linear_svm_building_footprint.joblib")
    tuning_df = pd.DataFrame(rows)
    tuning_df.to_csv(output_paths["metrics"] / "svm_tuning.csv", index=False)
    return best_model, tuning_df, best_c


def _dice_loss_from_logits(logits: Any, targets: Any) -> Any:
    import torch

    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    denom = torch.sum(probs, dims) + torch.sum(targets, dims)
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def build_unet_model() -> Any:
    import torch
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class UNetSmall(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down1 = DoubleConv(3, 32)
            self.pool1 = nn.MaxPool2d(2)
            self.down2 = DoubleConv(32, 64)
            self.pool2 = nn.MaxPool2d(2)
            self.down3 = DoubleConv(64, 128)
            self.pool3 = nn.MaxPool2d(2)
            self.bridge = DoubleConv(128, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.conv3 = DoubleConv(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.conv2 = DoubleConv(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.conv1 = DoubleConv(64, 32)
            self.head = nn.Conv2d(32, 1, kernel_size=1)

        def forward(self, x: Any) -> Any:
            d1 = self.down1(x)
            d2 = self.down2(self.pool1(d1))
            d3 = self.down3(self.pool2(d2))
            bridge = self.bridge(self.pool3(d3))
            u3 = self.up3(bridge)
            u3 = self.conv3(torch.cat([u3, d3], dim=1))
            u2 = self.up2(u3)
            u2 = self.conv2(torch.cat([u2, d2], dim=1))
            u1 = self.up1(u2)
            u1 = self.conv1(torch.cat([u1, d1], dim=1))
            return self.head(u1)

    return UNetSmall()


def select_torch_device(prefer_cuda: bool = True) -> Any:
    import torch
    import torch.nn as nn

    if prefer_cuda and torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            probe = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True),
            ).to(device)
            x = torch.randn(1, 3, 16, 16, device=device)
            with torch.no_grad():
                _ = probe(x)
            torch.cuda.synchronize()
            name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            print(f"[Torch] Using CUDA device: {name}, capability={capability}")
            return device
        except Exception as exc:
            print("[Torch][WARN] CUDA is visible but failed a runtime kernel probe.")
            print(f"[Torch][WARN] Falling back to CPU. Original CUDA error: {type(exc).__name__}: {exc}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    print("[Torch] Using CPU device.")
    return torch.device("cpu")


def _augment_patch(patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        patch = np.flip(patch, axis=1)
        mask = np.flip(mask, axis=1)
    if random.random() < 0.5:
        patch = np.flip(patch, axis=0)
        mask = np.flip(mask, axis=0)
    k = random.randint(0, 3)
    if k:
        patch = np.rot90(patch, k)
        mask = np.rot90(mask, k)
    if random.random() < 0.8:
        patch_f = patch.astype(np.float32)
        contrast = random.uniform(0.9, 1.1)
        brightness = random.uniform(-10, 10)
        patch = np.clip((patch_f - 127.5) * contrast + 127.5 + brightness, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(patch), np.ascontiguousarray(mask)


class PatchSegmentationDataset:
    def __init__(self, records: list[dict[str, Any]], cfg: ExperimentConfig, augment: bool) -> None:
        self.records = records
        self.cfg = cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch

        patch, mask = read_patch(self.records[index])
        patch = preprocess_patch(patch, self.cfg)
        if self.augment:
            patch, mask = _augment_patch(patch, mask)
        image = patch.astype(np.float32) / 255.0
        target = (mask > 127).astype(np.float32)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(target[None, :, :]).float()
        return image_tensor, mask_tensor


def make_unet_dataset(records: list[dict[str, Any]], cfg: ExperimentConfig, augment: bool) -> Any:
    import torch

    if not hasattr(torch, "from_numpy"):
        raise RuntimeError("PyTorch is required for the U-Net dataset.")
    return PatchSegmentationDataset(records, cfg, augment)


def train_unet(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    cfg: ExperimentConfig,
) -> tuple[Any, float, pd.DataFrame]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = select_torch_device(cfg.unet_use_cuda)
    set_global_seed(cfg.random_seed)
    model = build_unet_model().to(device)
    train_dataset = make_unet_dataset(train_records, cfg, augment=True)
    val_dataset = make_unet_dataset(val_records, cfg, augment=False)
    num_workers = 0 if os.name == "nt" else cfg.unet_num_workers
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.unet_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.unet_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.unet_learning_rate,
        weight_decay=cfg.unet_weight_decay,
    )

    output_paths = ensure_output_dirs(cfg)
    best_path = output_paths["models"] / "unet_best.pth"
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, cfg.unet_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for images, masks in tqdm(train_loader, desc=f"U-Net train epoch {epoch}"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks) + _dice_loss_from_logits(logits, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"U-Net val epoch {epoch}"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, masks) + _dice_loss_from_logits(logits, masks)
                val_losses.append(float(loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "device": str(device),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_paths["metrics"] / "unet_training_history.csv", index=False)

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.unet_patience:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = tune_unet_threshold(model, val_records, cfg, device)
    return model, threshold, pd.DataFrame(history)


def predict_unet_patch(
    patch: np.ndarray,
    model: Any,
    cfg: ExperimentConfig,
    threshold: float,
    device: Any | None = None,
) -> np.ndarray:
    import torch

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    processed = preprocess_patch(patch, cfg).astype(np.float32) / 255.0
    tensor = torch.from_numpy(processed.transpose(2, 0, 1)[None]).float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
    raw = np.where(prob >= threshold, 255, 0).astype(np.uint8)
    return morphological_cleanup(raw, cfg) if cfg.unet_apply_morphology else raw


def tune_unet_threshold(model: Any, val_records: list[dict[str, Any]], cfg: ExperimentConfig, device: Any) -> float:
    records = val_records
    if cfg.max_unet_val_patches_for_threshold is not None:
        records = val_records[: cfg.max_unet_val_patches_for_threshold]
    rows: list[dict[str, Any]] = []
    best_threshold = cfg.unet_threshold_grid[0]
    best_dice = -1.0
    for threshold in cfg.unet_threshold_grid:
        df = evaluate_method(
            f"U-Net_thr_{threshold}",
            records,
            lambda patch, t=threshold: predict_unet_patch(patch, model, cfg, t, device),
            cfg,
        )
        dice = float(df["dice"].mean())
        rows.append({"threshold": threshold, "val_dice": dice})
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold

    output_paths = ensure_output_dirs(cfg)
    pd.DataFrame(rows).to_csv(output_paths["metrics"] / "unet_threshold_tuning.csv", index=False)
    with (output_paths["metrics"] / "unet_selected_threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": best_threshold, "val_dice": best_dice}, f, indent=2)
    return float(best_threshold)


def make_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()
    pred_b = binary_mask(pred)
    gt_b = binary_mask(gt)
    colors = {
        "tp": np.array([0, 220, 0], dtype=np.uint8),
        "fp": np.array([255, 40, 40], dtype=np.uint8),
        "fn": np.array([40, 80, 255], dtype=np.uint8),
    }
    masks = {
        "tp": np.logical_and(pred_b, gt_b),
        "fp": np.logical_and(pred_b, ~gt_b),
        "fn": np.logical_and(~pred_b, gt_b),
    }
    for key, mask in masks.items():
        overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * colors[key]).astype(np.uint8)
    return overlay


def save_qualitative_grids(
    records: list[dict[str, Any]],
    predictors: dict[str, Callable[[np.ndarray], np.ndarray]],
    cfg: ExperimentConfig,
) -> None:
    output_paths = ensure_output_dirs(cfg)
    selected = records[: cfg.qualitative_examples]
    for idx, record in enumerate(selected):
        patch, gt = read_patch(record)
        columns = [("Image", patch), ("Ground truth", np.repeat(gt[:, :, None], 3, axis=2))]
        for method, predictor in predictors.items():
            pred = predictor(patch)
            columns.append((method, make_overlay(patch, pred, gt)))

        fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
        if len(columns) == 1:
            axes = [axes]
        for ax, (title, image) in zip(axes, columns):
            ax.imshow(image)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{record['patch_id']} | green=TP, red=FP, blue=FN", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_paths["figures"] / f"qualitative_grid_{idx:03d}.png", dpi=160)
        plt.close(fig)


def plot_method_comparison(summary_df: pd.DataFrame, cfg: ExperimentConfig) -> None:
    output_paths = ensure_output_dirs(cfg)
    metrics = ["iou", "dice", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(summary_df))
    for idx, row in summary_df.iterrows():
        values = [row[f"{metric}_mean"] for metric in metrics]
        ax.bar(x + idx * width, values, width=width, label=row["method"])
    ax.set_xticks(x + width * (len(summary_df) - 1) / 2)
    ax.set_xticklabels([m.upper() if m == "f1" else m.title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Building footprint segmentation metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_paths["figures"] / "method_comparison.png", dpi=180)
    plt.close(fig)


def write_report_snippets(summary_df: pd.DataFrame, cfg: ExperimentConfig) -> None:
    output_paths = ensure_output_dirs(cfg)
    table_path = output_paths["reports"] / "final_metrics_table.tex"
    summary_path = output_paths["reports"] / "final_metrics_summary.tex"

    def pm(row: pd.Series, metric: str, digits: int = 3) -> str:
        return f"{row[f'{metric}_mean']:.{digits}f} $\\pm$ {row[f'{metric}_std']:.{digits}f}"

    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & IoU & Dice & Precision & Recall & Count Err. & Area Err. \\\\",
        "\\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['method']} & {pm(row, 'iou')} & {pm(row, 'dice')} & "
            f"{pm(row, 'precision')} & {pm(row, 'recall')} & "
            f"{row['count_err_mean']:.2f} & {row['area_abs_error_mean']:.2f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = summary_df.sort_values("dice_mean", ascending=False).iloc[0]
    summary_text = (
        f"Best method by Dice: \\textbf{{{best['method']}}} "
        f"with Dice={best['dice_mean']:.3f} and IoU={best['iou_mean']:.3f}."
    )
    summary_path.write_text(summary_text + "\n", encoding="utf-8")


def run_classical_baselines(test_records: list[dict[str, Any]], cfg: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, Callable]]:
    predictors = {
        "K-Means": lambda patch: morphological_cleanup(segment_kmeans(preprocess_patch(patch, cfg), cfg), cfg),
        "Otsu": lambda patch: morphological_cleanup(segment_otsu(preprocess_patch(patch, cfg), cfg), cfg),
    }
    frames = [evaluate_method(name, test_records, predictor, cfg) for name, predictor in predictors.items()]
    return pd.concat(frames, ignore_index=True), predictors


def run_full_experiment(
    cfg: ExperimentConfig,
    run_classical: bool = True,
    run_svm_model: bool = True,
    run_unet_model: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_paths, records_by_split = prepare_manifests(cfg)
    test_records = records_by_split["test"]
    all_frames: list[pd.DataFrame] = []
    qualitative_predictors: dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    if run_classical:
        classical_df, classical_predictors = run_classical_baselines(test_records, cfg)
        all_frames.append(classical_df)
        qualitative_predictors.update(classical_predictors)

    if run_svm_model:
        svm_model, tuning_df, best_c = tune_svm(records_by_split["train"], records_by_split["val"], cfg)
        svm_df = evaluate_method(
            "SVM",
            test_records,
            lambda patch: predict_svm_patch(patch, svm_model, cfg),
            cfg,
        )
        svm_df["selected_c"] = best_c
        tuning_df.to_csv(output_paths["metrics"] / "svm_tuning.csv", index=False)
        all_frames.append(svm_df)
        qualitative_predictors["SVM"] = lambda patch: predict_svm_patch(patch, svm_model, cfg)

    if run_unet_model:
        try:
            unet_model, threshold, history_df = train_unet(records_by_split["train"], records_by_split["val"], cfg)
            device = next(unet_model.parameters()).device
            history_df.to_csv(output_paths["metrics"] / "unet_training_history.csv", index=False)
            unet_df = evaluate_method(
                "U-Net",
                test_records,
                lambda patch: predict_unet_patch(patch, unet_model, cfg, threshold, device),
                cfg,
            )
            unet_df["threshold"] = threshold
            all_frames.append(unet_df)
            qualitative_predictors["U-Net"] = lambda patch: predict_unet_patch(patch, unet_model, cfg, threshold, device)
        except ImportError as exc:
            print(f"[Skip U-Net] PyTorch is not available: {exc}")

    if not all_frames:
        raise RuntimeError("No experiment branch was selected.")

    per_patch_df = pd.concat(all_frames, ignore_index=True)
    summary_df = aggregate_summary(per_patch_df)
    per_patch_df.to_csv(output_paths["metrics"] / "per_patch_metrics.csv", index=False)
    summary_df.to_csv(output_paths["metrics"] / "final_summary.csv", index=False)
    plot_method_comparison(summary_df, cfg)
    write_report_snippets(summary_df, cfg)
    save_qualitative_grids(test_records, qualitative_predictors, cfg)
    return per_patch_df, summary_df


def run_metric_self_tests() -> None:
    cfg = ExperimentConfig()
    empty = np.zeros((16, 16), dtype=np.uint8)
    full = np.ones((16, 16), dtype=np.uint8) * 255
    half = empty.copy()
    half[:, :8] = 255

    perfect = evaluate_single(full, full)
    assert perfect["iou"] == 1.0 and perfect["dice"] == 1.0
    empty_perfect = evaluate_single(empty, empty)
    assert empty_perfect["iou"] == 1.0 and empty_perfect["dice"] == 1.0
    missed = evaluate_single(empty, full)
    assert missed["iou"] == 0.0 and missed["recall"] == 0.0
    partial = evaluate_single(half, full)
    assert 0.0 < partial["iou"] < 1.0

    cleaned = morphological_cleanup(full, cfg)
    assert cleaned.shape == full.shape
    print("Metric and morphology self-tests passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final Project 16 building footprint experiment")
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to AerialImageDataset")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-svm", action="store_true")
    parser.add_argument("--skip-unet", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use tiny patch limits for debugging")
    parser.add_argument("--self-test", action="store_true", help="Run metric self-tests only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_metric_self_tests()
        return

    cfg = ExperimentConfig(dataset_root=args.dataset_root, output_dir=args.output_dir)
    if args.quick:
        cfg.max_train_patches = 24
        cfg.max_val_patches = 8
        cfg.max_test_patches = 8
        cfg.max_train_pixels = 20_000
        cfg.unet_epochs = 1
        cfg.unet_batch_size = 2
        cfg.max_svm_val_patches_for_tuning = 4
        cfg.max_unet_val_patches_for_threshold = 4
        cfg.qualitative_examples = 2

    run_full_experiment(
        cfg,
        run_classical=not args.skip_classical,
        run_svm_model=not args.skip_svm,
        run_unet_model=not args.skip_unet,
    )


if __name__ == "__main__":
    main()
