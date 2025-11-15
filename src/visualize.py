"""Visualization helpers for kernels, feature maps, and saliency."""
from __future__ import annotations

from typing import Tuple

import matplotlib.cm as cm
import numpy as np
import torch


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_val = arr.min()
    max_val = arr.max()
    if np.isclose(max_val - min_val, 0.0):
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3:
        array = array[0]
    norm = _normalize(array)
    return (norm * 255).astype(np.uint8)


def render_feature_grid(tensor: torch.Tensor,
                        max_items: int = 16,
                        num_cols: int = 4,
                        pad: int = 2) -> np.ndarray:
    maps = tensor.detach().cpu().numpy()
    if maps.ndim == 4:
        maps = maps[0]
    count = min(max_items, maps.shape[0])
    maps = maps[:count]
    num_cols = min(num_cols, count)
    num_rows = int(np.ceil(count / num_cols))
    h, w = maps.shape[1:]
    canvas = np.ones((num_rows * h + pad * (num_rows - 1), num_cols * w + pad * (num_cols - 1)), dtype=np.float32)
    for idx in range(count):
        row = idx // num_cols
        col = idx % num_cols
        top = row * (h + pad)
        left = col * (w + pad)
        tile = _normalize(maps[idx])
        canvas[top:top + h, left:left + w] = tile
    return (canvas * 255).astype(np.uint8)


def render_kernel_grid(weights: torch.Tensor,
                       num_cols: int = 6,
                       pad: int = 2) -> np.ndarray:
    weight = weights.detach().cpu().numpy()
    out_channels = weight.shape[0]
    count = out_channels
    num_cols = min(num_cols, count)
    num_rows = int(np.ceil(count / num_cols))
    kernel_size = weight.shape[-1]
    canvas = np.ones((num_rows * kernel_size + pad * (num_rows - 1),
                      num_cols * kernel_size + pad * (num_cols - 1)), dtype=np.float32)
    for idx in range(count):
        row = idx // num_cols
        col = idx % num_cols
        top = row * (kernel_size + pad)
        left = col * (kernel_size + pad)
        tile = weight[idx, 0]
        canvas[top:top + kernel_size, left:left + kernel_size] = _normalize(tile)
    return (canvas * 255).astype(np.uint8)


def overlay_saliency(base_image: np.ndarray,
                     saliency: np.ndarray,
                     alpha: float = 0.6) -> np.ndarray:
    base = _normalize(base_image)
    heat = _normalize(saliency)
    cmap = cm.get_cmap("magma")
    heat_rgb = cmap(heat)[..., :3]
    base_rgb = np.stack([base] * 3, axis=-1)
    overlay = (1 - alpha) * base_rgb + alpha * heat_rgb
    overlay = np.clip(overlay, 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def probs_to_topk(probs: torch.Tensor, topk: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    values, indices = torch.topk(probs, k=topk, dim=1)
    return indices.squeeze(0).cpu().numpy(), values.squeeze(0).cpu().numpy()
