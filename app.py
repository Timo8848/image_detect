from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.transform import resize as sk_resize
from streamlit_drawable_canvas import st_canvas
from torchvision import datasets

from src import gestures, preprocess, visualize
from src.inference import MNIST_CLASSES, compute_saliency, forward_with_artifacts
from src.model import load_lenet

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "lenet_mnist.pt"
DATA_DIR = APP_DIR / "data"
CANVAS_PX = 420
DRAWING_RES_OPTIONS = [28, 140]
KERNEL_OPTIONS = [3, 5]
FILTER_OPTIONS = [6, 12, 16]
STRIDE_OPTIONS = [1, 2]
PADDING_OPTIONS = [0, 1, 2]
EXPR_HISTORY_LIMIT = 8

LAYER_EXPLANATIONS = {
    "conv": "Convolution slides a small kernel across the image. The kernel size controls how big the patch is. Stride tells us how far we move the kernel each step, while padding adds a border of zeros so edges get seen.",
    "activation": "ReLU keeps positive activations and discards negatives, so only strong evidence flows forward.",
    "pool": "Pooling downsamples feature maps. Max pooling keeps the strongest activation inside each window, making the model less sensitive to pixel shifts.",
    "reshape": "Flattening rearranges spatial features into a vector, preparing them for fully-connected layers.",
    "dense": "Dense layers mix every input activation with learned weights. They assemble the high-level pieces into class evidence.",
    "softmax": "Softmax turns logits into probabilities that sum to 1. The largest probability is the predicted digit.",
}


@st.cache_resource(show_spinner=False)
def load_cached_model(random_mode: bool):
    weights = str(MODEL_PATH) if not random_mode else None
    bundle = load_lenet(weights_path=weights, strict=not random_mode)
    return bundle.model


@st.cache_resource(show_spinner=False)
def load_mnist_samples(split: str = "test"):
    dataset = datasets.MNIST(root=str(DATA_DIR), train=(split == "train"), download=True)
    images = dataset.data.numpy()
    labels = dataset.targets.numpy()
    return images, labels


def _ensure_state():
    defaults = {
        "canvas_history": [],
        "canvas_bg": None,
        "canvas_key": 0,
        "preprocess": None,
        "current_step": 0,
        "inference": None,
        "saliency": None,
        "model_mode": "Pretrained",
        "expression_tokens": [],
        "calc_result": None,
        "calc_error": None,
        "calc_history": [],
        "last_eval_expr": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _rgba_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.astype(np.uint8))


def _resize_rgba(image: np.ndarray, target: int) -> np.ndarray:
    if image.shape[0] == target:
        return image
    scaled = sk_resize(image, (target, target, image.shape[2]), preserve_range=True, anti_aliasing=True)
    return scaled.astype(np.uint8)


def _downsample_for_mode(image: Optional[np.ndarray], resolution: int) -> Optional[np.ndarray]:
    if image is None:
        return None
    if image.shape[0] == resolution:
        return image
    resized = sk_resize(image, (resolution, resolution, image.shape[2]), preserve_range=True, anti_aliasing=True)
    return resized.astype(np.uint8)


def _random_mnist_background(resolution: int) -> tuple[np.ndarray, int]:
    images, labels = load_mnist_samples()
    idx = random.randrange(len(images))
    digit = images[idx] / 255.0
    rgba = preprocess.prepare_mnist_example(digit, target_size=resolution)
    bg = _resize_rgba(rgba, CANVAS_PX)
    return bg, int(labels[idx])


def _conv_controls() -> Dict[str, int]:
    with st.expander("⚙️ Convolution controls", expanded=True):
        kernel = st.radio("Kernel size", KERNEL_OPTIONS, index=1,
                          format_func=lambda k: f"{k}×{k}",
                          help="3×3 sees a tiny patch; 5×5 sees more context.")
        stride = st.radio("Stride", STRIDE_OPTIONS, index=0,
                          help="Stride is how far the kernel jumps each move.")
        padding = st.radio("Padding", PADDING_OPTIONS, index=0,
                            help="Padding adds zeros around the edge so corners are covered.")
        filters = st.radio("# Filters", FILTER_OPTIONS, index=len(FILTER_OPTIONS) - 1,
                           help="More filters learn more patterns (edges, loops, etc.) in layer 1.")
    return {"kernel_size": kernel, "stride": stride, "padding": padding, "filters": filters}


def _model_controls() -> str:
    mode = st.radio("Model mode", ["Pretrained", "Random init"], horizontal=True,
                    help="Random init demonstrates how untrained weights behave.",
                    key="model_mode_selector")
    if mode == "Random init" and st.button("Reshuffle random weights"):
        load_cached_model.clear()
        st.success("Random weights reinitialized.")
    st.session_state["model_mode"] = mode
    return mode


def _render_sidebar(layer_kind: Optional[str], layer_name: Optional[str], meta: Optional[Dict[str, int]]) -> None:
    st.sidebar.header("Explain mode")
    if layer_kind is None:
        st.sidebar.info("Draw a digit and run inference to unlock the walkthrough.")
        return
    st.sidebar.subheader(layer_name)
    st.sidebar.write(LAYER_EXPLANATIONS.get(layer_kind, ""))
    if meta:
        meta_lines = [f"**{k.capitalize()}**: {v}" for k, v in meta.items()]
        st.sidebar.markdown("\n".join(meta_lines))
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Stride = jump size · Padding = zero-border thickness · Pooling = compressing activations"
    )


def _plot_topk(result) -> tuple[str, float]:
    indices, values = visualize.probs_to_topk(result.probs, topk=3)
    labels = [MNIST_CLASSES[int(i)] for i in indices]
    data = pd.DataFrame({"Digit": labels, "Probability": values})
    st.bar_chart(data.set_index("Digit"))
    return labels[0], float(values[0])


def _expression_tokens() -> list[str]:
    return st.session_state["expression_tokens"]


def _record_history(expression: str, result: float) -> None:
    history = st.session_state["calc_history"]
    history.insert(0, {"Expression": expression, "Result": result})
    if len(history) > EXPR_HISTORY_LIMIT:
        history.pop()


def _append_symbol(prediction: gestures.SymbolPrediction) -> Optional[str]:
    tokens = _expression_tokens()
    symbol = prediction.label
    if prediction.kind == "operator":
        if not tokens:
            return "Start with a digit before adding an operator."
        if tokens[-1] in gestures.OPERATOR_SYMBOLS:
            return "Two operators cannot be adjacent."
    tokens.append(symbol)
    return None


def _pop_symbol() -> None:
    tokens = _expression_tokens()
    if tokens:
        tokens.pop()


def _clear_expression() -> None:
    st.session_state["expression_tokens"] = []
    st.session_state["calc_result"] = None
    st.session_state["calc_error"] = None
    st.session_state["last_eval_expr"] = None


def main() -> None:
    st.set_page_config(page_title="Interactive CNN Digit Classifier", layout="wide")
    _ensure_state()

    st.title("Interactive CNN Digit Classifier")
    st.caption("Draw a digit, peek inside every layer, and see why the network chose its answer.")

    left, right = st.columns([2, 1])
    with right:
        model_mode = _model_controls()
        conv_options = _conv_controls()
    with left:
        drawing_resolution = st.radio("Canvas resolution", options=DRAWING_RES_OPTIONS, index=1,
                                      format_func=lambda x: f"{x}×{x}", horizontal=True,
                                      help="Draw at a coarse 28×28 grid or a finer 140×140 grid.")
        bg_image = st.session_state.get("canvas_bg")
        pil_bg = _rgba_to_pil(bg_image) if isinstance(bg_image, np.ndarray) else None
        canvas_result = st_canvas(
            stroke_width=st.slider("Brush size", 4, 40, 18),
            stroke_color="#000000",
            background_color="#FFFFFF",
            background_image=pil_bg,
            update_streamlit=True,
            height=CANVAS_PX,
            width=CANVAS_PX,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state['canvas_key']}",
        )

        canvas_image = canvas_result.image_data if canvas_result is not None else None
        history = st.session_state["canvas_history"]
        if canvas_image is not None:
            canvas_uint8 = canvas_image.astype(np.uint8)
            st.session_state["last_canvas"] = canvas_uint8
            if not history or not np.array_equal(history[-1], canvas_uint8):
                history.append(canvas_uint8.copy())
                if len(history) > 12:
                    history.pop(0)
        else:
            canvas_uint8 = st.session_state.get("last_canvas")

        btn_cols = st.columns(3)
        with btn_cols[0]:
            if st.button("Clear"):
                st.session_state["canvas_history"] = []
                st.session_state["canvas_bg"] = None
                st.session_state["last_canvas"] = None
                st.session_state["canvas_key"] += 1
        with btn_cols[1]:
            if st.button("Undo") and st.session_state["canvas_history"]:
                history = st.session_state["canvas_history"]
                if history:
                    history.pop()
                prev = history[-1] if history else None
                st.session_state["canvas_bg"] = prev
                st.session_state["last_canvas"] = prev
                st.session_state["canvas_key"] += 1
        with btn_cols[2]:
            if st.button("Random sample"):
                bg, label = _random_mnist_background(drawing_resolution)
                st.session_state["canvas_bg"] = bg.copy()
                st.session_state["canvas_history"] = [bg.copy()]
                st.session_state["last_canvas"] = bg.copy()
                st.session_state["random_sample_label"] = label
                st.session_state["canvas_key"] += 1
                st.toast(f"Loaded MNIST digit {label}")

        downsampled = _downsample_for_mode(canvas_uint8, drawing_resolution) if canvas_uint8 is not None else None
        st.caption(f"Downsampled canvas ({drawing_resolution}×{drawing_resolution})")
        if downsampled is not None:
            st.image(downsampled.astype(np.uint8), use_column_width=True)

        if st.button("Classify", type="primary"):
            if downsampled is None:
                st.warning("Draw on the canvas first.")
            else:
                prep = preprocess.preprocess_canvas(downsampled)
                st.session_state["preprocess"] = prep
                st.session_state["current_step"] = 0
                st.session_state["layer_slider"] = 0

    prep_result = st.session_state.get("preprocess")
    inference_result = None
    saliency_map = None
    sidebar_payload = (None, None, None)
    if prep_result:
        model = load_cached_model(random_mode=(model_mode == "Random init"))
        inference_result = forward_with_artifacts(model, prep_result.tensor, conv1_overrides=conv_options)
        saliency_tensor = compute_saliency(model, prep_result.tensor, conv1_overrides=conv_options)
        saliency_map = saliency_tensor.numpy()
    else:
        st.info("Start by drawing a digit and clicking Classify.")

    gesture_prediction = None
    if inference_result and prep_result is not None:
        probs_tensor = inference_result.probs[0] if inference_result.probs.ndim == 2 else inference_result.probs
        gesture_prediction = gestures.classify_gesture(prep_result.processed_image, probs_tensor)

    if inference_result:
        st.subheader("Prediction readout")
        top_label, top_prob = _plot_topk(inference_result)
        st.metric("Predicted digit", top_label, delta=f"{top_prob * 100:.1f}% confidence")
        if prep_result and saliency_map is not None:
            overlay = visualize.overlay_saliency(prep_result.processed_image, saliency_map)
            st.image(overlay, caption="Saliency overlay (brighter = more influence)")

        artifacts = inference_result.layer_artifacts
        total_steps = len(artifacts)
        st.session_state["current_step"] = min(st.session_state["current_step"], total_steps - 1)
        controls = st.columns([1, 1, 3])
        with controls[0]:
            if st.button("Single step ▶"):
                st.session_state["current_step"] = min(total_steps - 1, st.session_state["current_step"] + 1)
        with controls[1]:
            if st.button("Play all ⏭"):
                st.session_state["current_step"] = total_steps - 1
        with controls[2]:
            st.session_state["current_step"] = st.slider(
                "Scrub layers",
                0,
                total_steps - 1,
                value=st.session_state["current_step"],
                key="layer_slider",
            )

        current_artifact = artifacts[st.session_state["current_step"]]
        sidebar_payload = (current_artifact.kind, current_artifact.name, current_artifact.meta)

        st.subheader(f"Layer {st.session_state['current_step'] + 1}/{total_steps}: {current_artifact.name}")
        info_cols = st.columns(2)
        with info_cols[0]:
            st.caption("Layer input")
            st.image(visualize.tensor_to_image(current_artifact.input))
        with info_cols[1]:
            st.caption("Layer output")
            if current_artifact.output.ndim >= 3:
                st.image(visualize.render_feature_grid(current_artifact.output))
            else:
                vector = current_artifact.output.detach().cpu().numpy().flatten()
                df = pd.DataFrame({"Index": list(range(len(vector))), "Value": vector})
                st.bar_chart(df.set_index("Index"))

        if current_artifact.weights is not None and current_artifact.weights.ndim == 4:
            st.caption("Conv kernels")
            st.image(visualize.render_kernel_grid(current_artifact.weights))

        timeline = pd.DataFrame(
            {
                "Layer": [a.name for a in artifacts],
                "Type": [a.kind for a in artifacts],
            }
        )
        st.dataframe(timeline, use_container_width=True, hide_index=True)

    if gesture_prediction:
        st.subheader("Gesture-driven calculator")
        st.caption("Add each gesture to the expression builder to evaluate handwritten equations.")
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Interpreted symbol", gesture_prediction.label,
                      delta=gesture_prediction.kind.capitalize())
        with info_cols[1]:
            st.metric("Symbol confidence", f"{gesture_prediction.confidence:.2f}")
        with info_cols[2]:
            st.metric("Digit confidence", f"{gesture_prediction.digit_prob:.2f}")

        operator_scores = {
            symbol: gesture_prediction.operator_scores.get(symbol, 0.0) for symbol in gestures.OPERATOR_SYMBOLS
        }
        scores_df = pd.DataFrame(
            {"Operator": list(operator_scores.keys()), "Score": list(operator_scores.values())}
        )
        st.dataframe(scores_df, hide_index=True, use_container_width=True)

        tokens = _expression_tokens()
        expression_display = gestures.expression_from_tokens(tokens)
        st.caption("Current expression")
        st.code(expression_display if expression_display else "…", language="text")

        button_cols = st.columns(3)
        with button_cols[0]:
            if st.button("Add to expression", type="primary"):
                error = _append_symbol(gesture_prediction)
                if error:
                    st.warning(error)
        with button_cols[1]:
            st.button("Backspace", disabled=not tokens, on_click=_pop_symbol)
        with button_cols[2]:
            st.button("Clear expression", disabled=not tokens, on_click=_clear_expression)

        preview_value, preview_error = gestures.evaluate_tokens(tokens) if tokens else (None, "Expression is empty.")
        if preview_error:
            st.info(preview_error)
        elif preview_value is not None:
            st.success(f"Preview result: {preview_value}")

        eval_cols = st.columns([1, 1])
        with eval_cols[0]:
            if st.button("Evaluate expression", disabled=not tokens):
                if preview_error:
                    st.warning(preview_error)
                else:
                    expr = gestures.expression_from_tokens(tokens)
                    st.session_state["calc_result"] = float(preview_value)
                    st.session_state["calc_error"] = None
                    st.session_state["last_eval_expr"] = expr
                    _record_history(expr, float(preview_value))
                    st.success(f"{expr} = {preview_value}")
        with eval_cols[1]:
            last_expr = st.session_state.get("last_eval_expr")
            last_result = st.session_state.get("calc_result")
            if last_expr and last_result is not None:
                st.metric("Last result", f"{last_expr} = {last_result}")

        history = st.session_state["calc_history"]
        if history:
            st.caption("Recent calculations")
            hist_df = pd.DataFrame(history)
            st.dataframe(hist_df, hide_index=True, use_container_width=True)

    _render_sidebar(*sidebar_payload)


if __name__ == "__main__":
    main()
