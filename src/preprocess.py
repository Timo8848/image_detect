"""Utilities for preparing canvas input for the CNN."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from skimage.transform import resize

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass
class PreprocessResult:
    tensor: torch.Tensor
    processed_image: np.ndarray
    bbox: Optional[Tuple[int, int, int, int]]


def _shift_image_to_center(img: np.ndarray) -> np.ndarray:
    coords = np.argwhere(img > 0.0)
    if coords.size == 0:
        return img
    cy, cx = coords.mean(axis=0)
    target_center = np.array(img.shape) / 2.0
    shift = np.round(target_center - np.array([cy, cx])).astype(int)
    dy, dx = int(shift[0]), int(shift[1])
    result = np.zeros_like(img)
    h, w = img.shape
    y_start_src = max(0, -dy)
    y_end_src = min(h, h - dy)
    x_start_src = max(0, -dx)
    x_end_src = min(w, w - dx)
    y_start_dst = max(0, dy)
    y_end_dst = y_start_dst + (y_end_src - y_start_src)
    x_start_dst = max(0, dx)
    x_end_dst = x_start_dst + (x_end_src - x_start_src)
    result[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = img[y_start_src:y_end_src, x_start_src:x_end_src]
    return result


def _crop_to_bbox(img: np.ndarray, threshold: float = 0.05) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    mask = img > threshold
    if not mask.any():
        return img, None
    ys, xs = np.where(mask)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    cropped = img[ymin:ymax + 1, xmin:xmax + 1]
    return cropped, (int(ymin), int(xmin), int(ymax), int(xmax))


def preprocess_canvas(image_data: Optional[np.ndarray],
                      target_size: int = 28,
                      invert: bool = True) -> PreprocessResult:
    """Convert RGBA canvas data into a normalized tensor."""

    if image_data is None:
        blank = torch.zeros(1, 1, target_size, target_size, dtype=torch.float32)
        norm_blank = (blank - MNIST_MEAN) / MNIST_STD
        return PreprocessResult(tensor=norm_blank, processed_image=np.zeros((target_size, target_size)), bbox=None)

    if image_data.dtype != np.float32:
        data = image_data.astype(np.float32)
    else:
        data = image_data.copy()

    rgb = data[..., :3] / 255.0
    if image_data.shape[-1] == 4:
        alpha = data[..., 3] / 255.0
    else:
        alpha = np.ones_like(rgb[..., 0])

    gray = rgb.mean(axis=2)
    gray = gray * alpha + (1 - alpha)
    if invert:
        gray = 1.0 - gray

    cropped, bbox = _crop_to_bbox(gray)
    h, w = cropped.shape
    if h == 0 or w == 0:
        cropped = np.zeros((target_size, target_size), dtype=np.float32)
    scale = 20.0 / max(h, w) if max(h, w) > 0 else 1.0
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = resize(cropped, (new_h, new_w), mode="reflect", anti_aliasing=True)
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    canvas = _shift_image_to_center(canvas)

    tensor = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0).float()
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    return PreprocessResult(tensor=tensor, processed_image=canvas, bbox=bbox)


def prepare_mnist_example(image: np.ndarray, target_size: int = 140) -> np.ndarray:
    """Upscale a MNIST (28x28) digit for the Streamlit canvas."""

    if image.ndim != 2:
        raise ValueError("MNIST examples must be rank-2 arrays")
    scaled = resize(image, (target_size, target_size), mode="reflect", anti_aliasing=True)
    scaled = (1.0 - scaled)  # invert back to black strokes on white
    rgb = np.stack([scaled] * 3, axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    alpha = np.full((target_size, target_size, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)
