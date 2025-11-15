from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.inference import forward_with_artifacts
from src.model import load_lenet

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lenet_mnist.pt"
DATA_DIR = ROOT / "data"


def test_layer_shapes_cpu():
    bundle = load_lenet(weights_path=str(MODEL_PATH), device=torch.device("cpu"))
    dummy = torch.zeros(1, 1, 28, 28)
    result = forward_with_artifacts(bundle.model, dummy)
    assert result.logits.shape == (1, 10)
    conv1 = result.layer_artifacts[0]
    assert conv1.output.shape[-2:] == (24, 24)
    flatten = next(artifact for artifact in result.layer_artifacts if artifact.name == "Flatten")
    assert flatten.output.shape[1] == 32 * 4 * 4


def test_pretrained_accuracy_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    test_ds = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    bundle = load_lenet(weights_path=str(MODEL_PATH), device=torch.device("cpu"))
    model = bundle.model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            logits = model(data)
            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    acc = correct / total
    assert acc >= 0.98, f"Pretrained accuracy dropped to {acc:.4f}"
