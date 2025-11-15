"""Inference helpers and layer capture utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .model import LeNetClassifier

MNIST_CLASSES = [str(i) for i in range(10)]


def _to_int(value: int) -> int:
    if isinstance(value, (tuple, list)):
        return int(value[0])
    if value is None:
        return 0
    return int(value)


@dataclass
class LayerArtifact:
    name: str
    kind: str
    input: torch.Tensor
    output: torch.Tensor
    weights: Optional[torch.Tensor] = None
    meta: Optional[Dict[str, int]] = None


@dataclass
class InferenceResult:
    logits: torch.Tensor
    probs: torch.Tensor
    pred_idx: int
    layer_artifacts: List[LayerArtifact]


def default_conv1_options(model: LeNetClassifier) -> Dict[str, int]:
    conv1 = model.conv1
    return {
        "kernel_size": conv1.kernel_size[0],
        "stride": conv1.stride[0],
        "padding": conv1.padding[0],
        "filters": conv1.out_channels,
    }


def _sanitize_conv1_options(model: LeNetClassifier, overrides: Optional[Dict[str, int]]) -> Dict[str, int]:
    opts = default_conv1_options(model)
    if not overrides:
        return opts
    opts.update({k: int(v) for k, v in overrides.items() if v is not None})
    # Cap filters and kernel size to supported limits
    max_filters = model.conv1.out_channels
    opts["filters"] = max(1, min(max_filters, opts["filters"]))
    max_kernel = model.conv1.kernel_size[0]
    opts["kernel_size"] = max(1, min(max_kernel, opts["kernel_size"]))
    opts["stride"] = max(1, opts["stride"])
    opts["padding"] = max(0, opts["padding"])
    return opts


def _apply_first_conv(model: LeNetClassifier,
                      x: torch.Tensor,
                      options: Dict[str, int]) -> torch.Tensor:
    weight = model.conv1.weight
    bias = model.conv1.bias
    base_kernel = weight.shape[-1]
    target_kernel = options["kernel_size"]
    if target_kernel < base_kernel:
        trim = (base_kernel - target_kernel) // 2
        weight_slice = weight[:, :, trim:trim + target_kernel, trim:trim + target_kernel]
    elif target_kernel > base_kernel:
        pad_total = target_kernel - base_kernel
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        weight_slice = F.pad(weight, (pad_left, pad_right, pad_left, pad_right))
    else:
        weight_slice = weight
    weight_slice = weight_slice[:options["filters"]]
    bias_slice = bias[:options["filters"]] if bias is not None else None
    conv = F.conv2d(x, weight_slice, bias_slice, stride=options["stride"], padding=options["padding"])
    return conv, weight_slice, bias_slice


def _record(artifacts: List[LayerArtifact], name: str, kind: str,
            input_tensor: torch.Tensor, output_tensor: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            meta: Optional[Dict[str, int]] = None) -> None:
    artifacts.append(
        LayerArtifact(
            name=name,
            kind=kind,
            input=input_tensor.detach().cpu(),
            output=output_tensor.detach().cpu(),
            weights=weights.detach().cpu() if weights is not None else None,
            meta=meta,
        )
    )


def forward_with_artifacts(model: LeNetClassifier,
                           input_tensor: torch.Tensor,
                           conv1_overrides: Optional[Dict[str, int]] = None,
                           device: Optional[torch.device] = None,
                           capture: bool = True,
                           detach_outputs: bool = True) -> InferenceResult:
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    x = input_tensor.to(device) if input_tensor.device != device else input_tensor
    artifacts: List[LayerArtifact] = []

    conv1_opts = _sanitize_conv1_options(model, conv1_overrides)
    conv1_out, conv1_weights, _ = _apply_first_conv(model, x, conv1_opts)
    if capture:
        _record(artifacts, "Conv 1", "conv", x, conv1_out, conv1_weights, meta=conv1_opts)

    relu1 = F.relu(conv1_out)
    if capture:
        _record(artifacts, "ReLU 1", "activation", conv1_out, relu1)

    pool1 = model.pool(relu1)
    if capture:
        meta = {
            "kernel_size": _to_int(model.pool.kernel_size),
            "stride": _to_int(model.pool.stride or model.pool.kernel_size),
        }
        _record(artifacts, "Pool 1", "pool", relu1, pool1, meta=meta)

    conv2_out = model.conv2(pool1)
    if capture:
        _record(artifacts, "Conv 2", "conv", pool1, conv2_out, model.conv2.weight,
                meta={"kernel_size": model.conv2.kernel_size[0], "stride": model.conv2.stride[0], "padding": model.conv2.padding[0]})

    relu2 = F.relu(conv2_out)
    if capture:
        _record(artifacts, "ReLU 2", "activation", conv2_out, relu2)

    pool2 = model.pool(relu2)
    if capture:
        meta = {
            "kernel_size": _to_int(model.pool.kernel_size),
            "stride": _to_int(model.pool.stride or model.pool.kernel_size),
        }
        _record(artifacts, "Pool 2", "pool", relu2, pool2, meta=meta)

    adaptive = model.adaptive_pool(pool2)
    if capture:
        _record(artifacts, "Adaptive AvgPool", "pool", pool2, adaptive, meta={"output_size": 4})

    flat = torch.flatten(adaptive, 1)
    if capture:
        _record(artifacts, "Flatten", "reshape", adaptive, flat)

    fc1 = model.fc1(flat)
    if capture:
        _record(artifacts, "Dense 1", "dense", flat, fc1, model.fc1.weight)
    act1 = F.relu(fc1)
    if capture:
        _record(artifacts, "Dense 1 Activation", "activation", fc1, act1)
    drop1 = model.dropout(act1)

    fc2 = model.fc2(drop1)
    if capture:
        _record(artifacts, "Dense 2", "dense", drop1, fc2, model.fc2.weight)
    act2 = F.relu(fc2)
    if capture:
        _record(artifacts, "Dense 2 Activation", "activation", fc2, act2)
    drop2 = model.dropout(act2)

    logits = model.fc3(drop2)
    probs = torch.softmax(logits, dim=1)
    if capture:
        _record(artifacts, "Logits", "dense", drop2, logits, model.fc3.weight)
        _record(artifacts, "Softmax", "softmax", logits, probs)

    pred_idx = int(torch.argmax(probs, dim=1).item())
    logits_out = logits.detach().cpu() if detach_outputs else logits
    probs_out = probs.detach().cpu() if detach_outputs else probs
    return InferenceResult(logits=logits_out,
                           probs=probs_out,
                           pred_idx=pred_idx,
                           layer_artifacts=artifacts)


def compute_saliency(model: LeNetClassifier,
                     input_tensor: torch.Tensor,
                     conv1_overrides: Optional[Dict[str, int]] = None,
                     target_class: Optional[int] = None) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    tensor = input_tensor.clone().detach().to(device).requires_grad_(True)
    result = forward_with_artifacts(model,
                                    tensor,
                                    conv1_overrides=conv1_overrides,
                                    capture=False,
                                    detach_outputs=False)
    if target_class is None:
        target_class = result.pred_idx
    result.logits[0, target_class].backward()
    saliency = tensor.grad.detach().abs().cpu()
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency.squeeze()
