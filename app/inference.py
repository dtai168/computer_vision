from __future__ import annotations

import json
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import cv2
import joblib
import numpy as np

from src.building_footprint_final import (
    ExperimentConfig,
    binary_mask,
    build_unet_model,
    count_buildings,
    evaluate_single,
    make_overlay,
    morphological_cleanup,
    predict_svm_patch,
    predict_unet_patch,
    preprocess_patch,
    segment_kmeans,
    segment_otsu,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUPPORTED_METHODS = ("K-Means", "Otsu", "SVM", "U-Net")


class ModelUnavailableError(RuntimeError):
    """Raised when a selected model artifact cannot be loaded."""


@dataclass
class AppArtifacts:
    cfg: ExperimentConfig
    output_dir: Path
    svm_model: Any | None
    unet_model: Any | None
    unet_threshold: float
    device: Any | None
    svm_error: str | None = None
    unet_error: str | None = None

    def available_methods(self) -> list[str]:
        methods = ["K-Means", "Otsu"]
        if self.svm_model is not None:
            methods.append("SVM")
        if self.unet_model is not None:
            methods.append("U-Net")
        return methods

    def device_label(self) -> str:
        return str(self.device) if self.device is not None else "not loaded"


@dataclass
class PredictionResult:
    method: str
    mask: np.ndarray
    overlay: np.ndarray
    seconds: float
    metrics: dict[str, float]
    error_overlay: np.ndarray | None = None
    padded_shape: tuple[int, int] | None = None


def _config_from_file(output_dir: Path) -> ExperimentConfig:
    cfg = ExperimentConfig(output_dir=str(output_dir))
    config_path = output_dir / "experiment_config.json"
    if not config_path.exists():
        return cfg

    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    field_names = {field.name for field in fields(ExperimentConfig)}
    for key, value in raw.items():
        if key not in field_names or key in {"output_dir", "dataset_root", "candidate_base_dirs"}:
            continue
        current_value = getattr(cfg, key)
        if isinstance(current_value, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)
    return cfg


def _load_svm_model(output_dir: Path) -> tuple[Any | None, str | None]:
    model_path = output_dir / "models" / "linear_svm_building_footprint.joblib"
    if not model_path.exists():
        return None, f"Missing SVM model: {model_path}"
    try:
        return joblib.load(model_path), None
    except Exception as exc:  # pragma: no cover - message is surfaced in the GUI
        return None, f"Cannot load SVM model: {type(exc).__name__}: {exc}"


def _load_unet_threshold(output_dir: Path) -> float:
    threshold_path = output_dir / "metrics" / "unet_selected_threshold.json"
    if not threshold_path.exists():
        return 0.5
    with threshold_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("threshold", 0.5))


def _load_unet_model(output_dir: Path) -> tuple[Any | None, float, Any | None, str | None]:
    model_path = output_dir / "models" / "unet_best.pth"
    threshold = _load_unet_threshold(output_dir)
    if not model_path.exists():
        return None, threshold, None, f"Missing U-Net model: {model_path}"

    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_unet_model()
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, threshold, device, None
    except Exception as exc:  # pragma: no cover - message is surfaced in the GUI
        return None, threshold, None, f"Cannot load U-Net model: {type(exc).__name__}: {exc}"


def load_artifacts(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> AppArtifacts:
    output_path = Path(output_dir)
    cfg = _config_from_file(output_path)
    cfg.output_dir = str(output_path)
    svm_model, svm_error = _load_svm_model(output_path)
    unet_model, threshold, device, unet_error = _load_unet_model(output_path)
    return AppArtifacts(
        cfg=cfg,
        output_dir=output_path,
        svm_model=svm_model,
        unet_model=unet_model,
        unet_threshold=threshold,
        device=device,
        svm_error=svm_error,
        unet_error=unet_error,
    )


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError("Expected a grayscale or RGB image array.")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.shape[2] != 3:
        raise ValueError("Expected an image with 1, 3, or 4 channels.")

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if np.issubdtype(image.dtype, np.floating):
        max_value = 1.0 if float(np.nanmax(image)) <= 1.0 else 255.0
        image = np.clip(image, 0.0, max_value) / max_value * 255.0
    else:
        image = np.clip(image, 0, 255)
    return np.ascontiguousarray(image.astype(np.uint8))


def normalize_mask(mask: np.ndarray, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = np.asarray(mask)
    if mask.dtype != np.uint8:
        if np.issubdtype(mask.dtype, np.floating):
            max_value = 1.0 if float(np.nanmax(mask)) <= 1.0 else 255.0
            mask = np.clip(mask, 0.0, max_value) / max_value * 255.0
        else:
            mask = np.clip(mask, 0, 255)
        mask = mask.astype(np.uint8)
    if target_shape is not None and mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def _pad_to_patch_grid(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    padded_height = max(patch_size, int(np.ceil(height / patch_size) * patch_size))
    padded_width = max(patch_size, int(np.ceil(width / patch_size) * patch_size))
    pad_bottom = padded_height - height
    pad_right = padded_width - width
    if pad_bottom == 0 and pad_right == 0:
        return image, (padded_height, padded_width)
    padded = cv2.copyMakeBorder(
        image,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )
    return padded, (padded_height, padded_width)


def _predict_patch(method: str, patch: np.ndarray, artifacts: AppArtifacts) -> np.ndarray:
    cfg = artifacts.cfg
    if method == "K-Means":
        processed = preprocess_patch(patch, cfg)
        return morphological_cleanup(segment_kmeans(processed, cfg), cfg)
    if method == "Otsu":
        processed = preprocess_patch(patch, cfg)
        return morphological_cleanup(segment_otsu(processed, cfg), cfg)
    if method == "SVM":
        if artifacts.svm_model is None:
            raise ModelUnavailableError(artifacts.svm_error or "SVM model is not loaded.")
        return predict_svm_patch(patch, artifacts.svm_model, cfg)
    if method == "U-Net":
        if artifacts.unet_model is None:
            raise ModelUnavailableError(artifacts.unet_error or "U-Net model is not loaded.")
        return predict_unet_patch(patch, artifacts.unet_model, cfg, artifacts.unet_threshold, artifacts.device)
    raise ValueError(f"Unsupported method: {method}")


def predict_mask(image: np.ndarray, method: str, artifacts: AppArtifacts) -> tuple[np.ndarray, tuple[int, int]]:
    image = ensure_rgb_uint8(image)
    cfg = artifacts.cfg
    patch_size = int(cfg.patch_size)
    padded, padded_shape = _pad_to_patch_grid(image, patch_size)
    output = np.zeros(padded.shape[:2], dtype=np.uint8)

    for y in range(0, padded.shape[0], patch_size):
        for x in range(0, padded.shape[1], patch_size):
            patch = padded[y : y + patch_size, x : x + patch_size]
            output[y : y + patch_size, x : x + patch_size] = _predict_patch(method, patch, artifacts)

    height, width = image.shape[:2]
    return output[:height, :width].astype(np.uint8), padded_shape


def prediction_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = ensure_rgb_uint8(image)
    mask_b = binary_mask(mask)
    overlay = image.copy()
    color = np.array([0, 220, 0], dtype=np.uint8)
    overlay[mask_b] = ((1 - alpha) * overlay[mask_b] + alpha * color).astype(np.uint8)
    return overlay


def summarize_without_ground_truth(mask: np.ndarray) -> dict[str, float]:
    mask_b = binary_mask(mask)
    return {
        "n_pred": float(count_buildings(mask)),
        "area_ratio_pred": float(mask_b.sum() / mask_b.size * 100.0),
    }


def run_inference(
    image: np.ndarray,
    method: str,
    artifacts: AppArtifacts,
    ground_truth: np.ndarray | None = None,
    overlay_alpha: float = 0.45,
) -> PredictionResult:
    image = ensure_rgb_uint8(image)
    start = time.perf_counter()
    mask, padded_shape = predict_mask(image, method, artifacts)
    seconds = time.perf_counter() - start

    gt_mask = normalize_mask(ground_truth, image.shape[:2]) if ground_truth is not None else None
    metrics = evaluate_single(mask, gt_mask) if gt_mask is not None else summarize_without_ground_truth(mask)
    return PredictionResult(
        method=method,
        mask=mask,
        overlay=prediction_overlay(image, mask, alpha=overlay_alpha),
        seconds=seconds,
        metrics=metrics,
        error_overlay=make_overlay(image, mask, gt_mask, alpha=overlay_alpha) if gt_mask is not None else None,
        padded_shape=padded_shape,
    )
