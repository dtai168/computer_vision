from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import SUPPORTED_METHODS, load_artifacts, run_inference


def main() -> None:
    artifacts = load_artifacts()
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    ground_truth = np.zeros((512, 512), dtype=np.uint8)
    ground_truth[120:360, 160:410] = 255

    for method in SUPPORTED_METHODS:
        result = run_inference(image, method, artifacts, ground_truth=ground_truth)
        assert result.mask.shape == image.shape[:2], (method, result.mask.shape)
        assert result.mask.dtype == np.uint8, (method, result.mask.dtype)
        assert result.overlay.shape == image.shape, (method, result.overlay.shape)
        assert result.error_overlay is not None, method
        assert "iou" in result.metrics and "dice" in result.metrics, method
        print(f"{method}: ok, seconds={result.seconds:.3f}, dice={result.metrics['dice']:.3f}")

    print("GUI inference smoke test passed.")


if __name__ == "__main__":
    main()
