"""Model definitions for the interactive CNN digit classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class LeNetClassifier(nn.Module):
    """LeNet-style network with a slightly wider first conv block."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        logits = self.fc3(x)
        return logits


@dataclass
class ModelBundle:
    """Convenience container for a model and metadata."""

    model: LeNetClassifier
    device: torch.device
    weights_path: Optional[Path]


def load_lenet(weights_path: Optional[str] = None,
               device: Optional[torch.device] = None,
               strict: bool = True) -> ModelBundle:
    """Load a LeNetClassifier, optionally from a checkpoint."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNetClassifier()
    model.to(device)
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return ModelBundle(model=model, device=device, weights_path=Path(weights_path) if weights_path else None)
