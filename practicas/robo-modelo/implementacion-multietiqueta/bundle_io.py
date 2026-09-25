from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_bundle(
    path: str | Path,
    *,
    images: torch.Tensor,
    teacher_probs: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    label_names: list[str] | tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    bundle = {
        "images": torch.as_tensor(images, dtype=torch.float32).cpu(),
        "metadata": metadata or {},
    }

    if teacher_probs is not None:
        bundle["teacher_probs"] = torch.as_tensor(teacher_probs, dtype=torch.float32).cpu()
    if labels is not None:
        bundle["labels"] = torch.as_tensor(labels, dtype=torch.float32).cpu()
    if label_names is not None:
        bundle["label_names"] = list(label_names)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return path


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = torch.load(Path(path), map_location="cpu")
    if "images" not in bundle:
        raise ValueError(f"Bundle {path} does not contain an 'images' tensor.")
    return bundle

