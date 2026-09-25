from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def predict_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            outputs.append(torch.sigmoid(model(batch)).cpu())

    return torch.cat(outputs, dim=0)


def format_label_set(binary_vector: np.ndarray | torch.Tensor, label_names: list[str]) -> str:
    if isinstance(binary_vector, torch.Tensor):
        binary_vector = binary_vector.detach().cpu().numpy()
    active = [name for name, value in zip(label_names, binary_vector) if value >= 0.5]
    return ", ".join(active) if active else "none"
