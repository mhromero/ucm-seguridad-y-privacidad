from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from data import IMAGE_SIZE, LABEL_NAMES


NUM_LABELS = len(LABEL_NAMES)


class TeacherCNN(nn.Module):
    """
    Simple multilabel CNN used as the teacher you will share with classmates.
    """

    def __init__(self, num_labels: int = NUM_LABELS) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, num_labels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))

    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(inputs))


class StudentCNN(nn.Module):
    """
    Student architecture family searched during extraction.
    """

    def __init__(
        self,
        *,
        conv_channels: Iterable[int] = (16, 32),
        classifier_hidden: int = 64,
        dropout: float = 0.15,
        num_labels: int = NUM_LABELS,
    ) -> None:
        super().__init__()

        channels = [1, *conv_channels]
        layers: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            sample = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)
            flattened_dim = int(self.features(sample).view(1, -1).shape[1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_labels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_student_from_config(config: dict[str, object]) -> StudentCNN:
    return StudentCNN(
        conv_channels=config["conv_channels"],
        classifier_hidden=int(config["classifier_hidden"]),
        dropout=float(config["dropout"]),
        num_labels=int(config.get("num_labels", NUM_LABELS)),
    )


def load_own_teacher_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[TeacherCNN, dict]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "Expected a checkpoint dictionary with a 'state_dict'. "
            "Use checkpoint_kind='serialized_module' or the peer adapter for external models."
        )

    model = TeacherCNN(num_labels=int(checkpoint.get("num_labels", NUM_LABELS))).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def load_serialized_module_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.to(device)
        checkpoint.eval()
        return checkpoint

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), nn.Module):
        model = checkpoint["model"].to(device)
        model.eval()
        return model

    raise ValueError(
        "The checkpoint is not a serialized torch.nn.Module. "
        "If your classmate sent a state_dict, implement peer_model_adapter.py with their architecture."
    )
