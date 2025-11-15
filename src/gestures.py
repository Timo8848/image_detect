"""Gesture heuristics and calculator helpers."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

OPERATOR_SYMBOLS = ["+", "-", "×", "÷"]
SYMBOL_TO_PY = {"+": "+", "-": "-", "×": "*", "÷": "/"}


@dataclass
class SymbolPrediction:
    label: str
    kind: str
    confidence: float
    digit_prob: float
    operator_scores: Dict[str, float]


def _draw_horizontal(canvas: np.ndarray, center: int, thickness: int) -> None:
    half = thickness // 2
    start = max(0, center - half)
    end = min(canvas.shape[0], center + half + 1)
    canvas[start:end, :] = 1.0


def _draw_vertical(canvas: np.ndarray, center: int, thickness: int) -> None:
    half = thickness // 2
    start = max(0, center - half)
    end = min(canvas.shape[1], center + half + 1)
    canvas[:, start:end] = 1.0


def _draw_diagonal(canvas: np.ndarray, direction: int, thickness: int) -> None:
    size = canvas.shape[0]
    half = max(0, thickness // 2)
    for y in range(size):
        x = y if direction > 0 else (size - 1 - y)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                yy = y + dy
                xx = x + dx
                if 0 <= yy < size and 0 <= xx < size:
                    canvas[yy, xx] = 1.0


def _draw_dot(canvas: np.ndarray, center: Tuple[int, int], radius: int) -> None:
    cy, cx = center
    for y in range(cy - radius, cy + radius + 1):
        if y < 0 or y >= canvas.shape[0]:
            continue
        for x in range(cx - radius, cx + radius + 1):
            if x < 0 or x >= canvas.shape[1]:
                continue
            if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                canvas[y, x] = 1.0


def _build_operator_templates(size: int = 28) -> Dict[str, np.ndarray]:
    canvas_templates: Dict[str, np.ndarray] = {}
    center = size // 2
    thickness = max(1, size // 14)
    diag_thickness = max(1, size // 18)
    dot_radius = max(1, size // 18)

    plus = np.zeros((size, size), dtype=np.float32)
    _draw_horizontal(plus, center, thickness)
    _draw_vertical(plus, center, thickness)
    canvas_templates["+"] = plus

    minus = np.zeros((size, size), dtype=np.float32)
    _draw_horizontal(minus, center, thickness)
    canvas_templates["-"] = minus

    times = np.zeros((size, size), dtype=np.float32)
    _draw_diagonal(times, direction=1, thickness=diag_thickness)
    _draw_diagonal(times, direction=-1, thickness=diag_thickness)
    canvas_templates["×"] = times

    divide = np.zeros((size, size), dtype=np.float32)
    _draw_horizontal(divide, center, thickness)
    _draw_dot(divide, (center - thickness * 2, center), dot_radius)
    _draw_dot(divide, (center + thickness * 2, center), dot_radius)
    canvas_templates["÷"] = divide

    return canvas_templates


OPERATOR_TEMPLATES = _build_operator_templates()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / denom)


def _score_operators(processed_image: np.ndarray) -> Dict[str, float]:
    normalized = processed_image.astype(np.float32)
    max_val = float(normalized.max())
    if max_val > 0:
        normalized /= max_val
    scores = {}
    for symbol, template in OPERATOR_TEMPLATES.items():
        scores[symbol] = _cosine_similarity(normalized, template)
    return scores


def classify_gesture(processed_image: np.ndarray,
                     digit_probs: torch.Tensor,
                     operator_threshold: float = 0.45,
                     preference_ratio: float = 1.1) -> SymbolPrediction:
    """Decide whether a gesture is a digit or operator using heuristics."""

    digit_tensor = digit_probs.detach().cpu().float()
    digit_prob, digit_idx = torch.max(digit_tensor, dim=0)
    digit_label = str(int(digit_idx))
    digit_conf = float(digit_prob.item())

    operator_scores = _score_operators(processed_image)
    operator_label = max(operator_scores, key=operator_scores.get)
    operator_conf = operator_scores[operator_label]

    use_operator = operator_conf >= operator_threshold and operator_conf * preference_ratio >= digit_conf

    if use_operator:
        return SymbolPrediction(
            label=operator_label,
            kind="operator",
            confidence=operator_conf,
            digit_prob=digit_conf,
            operator_scores=operator_scores,
        )

    return SymbolPrediction(
        label=digit_label,
        kind="digit",
        confidence=digit_conf,
        digit_prob=digit_conf,
        operator_scores=operator_scores,
    )


def expression_from_tokens(tokens: Sequence[str]) -> str:
    return "".join(tokens)


def tokens_to_python(tokens: Sequence[str]) -> str:
    return "".join(SYMBOL_TO_PY.get(token, token) for token in tokens)


def validate_tokens(tokens: Sequence[str]) -> Optional[str]:
    if not tokens:
        return "Expression is empty."
    if tokens[0] in OPERATOR_SYMBOLS:
        return "Expression cannot start with an operator."
    if tokens[-1] in OPERATOR_SYMBOLS:
        return "Expression cannot end with an operator."
    for left, right in zip(tokens, tokens[1:]):
        if left in OPERATOR_SYMBOLS and right in OPERATOR_SYMBOLS:
            return "Operators must be separated by digits."
    return None


def evaluate_tokens(tokens: Sequence[str]) -> Tuple[Optional[float], Optional[str]]:
    err = validate_tokens(tokens)
    if err:
        return None, err
    expression = tokens_to_python(tokens)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return None, f"Syntax error: {exc}"

    try:
        value = _eval_ast(tree.body)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Evaluation error: {exc}"
    return value, None


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}
_ALLOWED_UNARY = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"Unsupported operator: {op_type}")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _ALLOWED_BINOPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY:
            raise ValueError(f"Unsupported unary op: {op_type}")
        operand = _eval_ast(node.operand)
        return _ALLOWED_UNARY[op_type](operand)
    if isinstance(node, ast.Num):  # pragma: no cover - ast.Num for <3.8 compat
        return float(node.n)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant {node.value!r}")
    raise ValueError(f"Unsupported expression node: {type(node)}")
