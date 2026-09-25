from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset


IMAGE_SIZE = 32
LABEL_NAMES = ["circle", "square", "triangle", "cross"]
_ANCHORS = [(8, 8), (24, 8), (16, 16), (8, 24), (24, 24)]


def _draw_circle(draw: ImageDraw.ImageDraw, center: tuple[int, int], size: int, fill: int) -> None:
    x, y = center
    draw.ellipse((x - size, y - size, x + size, y + size), fill=fill)


def _draw_square(draw: ImageDraw.ImageDraw, center: tuple[int, int], size: int, fill: int) -> None:
    x, y = center
    draw.rectangle((x - size, y - size, x + size, y + size), fill=fill)


def _draw_triangle(draw: ImageDraw.ImageDraw, center: tuple[int, int], size: int, fill: int) -> None:
    x, y = center
    points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
    draw.polygon(points, fill=fill)


def _draw_cross(draw: ImageDraw.ImageDraw, center: tuple[int, int], size: int, fill: int) -> None:
    x, y = center
    thickness = max(2, size // 2)
    draw.rectangle((x - thickness, y - size, x + thickness, y + size), fill=fill)
    draw.rectangle((x - size, y - thickness, x + size, y + thickness), fill=fill)


_DRAWERS = [_draw_circle, _draw_square, _draw_triangle, _draw_cross]


def generate_shape_sample(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    draw = ImageDraw.Draw(image)

    num_shapes = int(rng.integers(1, 3))
    label_indices = rng.choice(len(LABEL_NAMES), size=num_shapes, replace=False)
    anchor_indices = rng.choice(len(_ANCHORS), size=num_shapes, replace=False)

    labels = np.zeros(len(LABEL_NAMES), dtype=np.float32)

    for label_index, anchor_index in zip(label_indices, anchor_indices):
        anchor_x, anchor_y = _ANCHORS[int(anchor_index)]
        jitter_x = int(rng.integers(-2, 3))
        jitter_y = int(rng.integers(-2, 3))
        center = (anchor_x + jitter_x, anchor_y + jitter_y)
        size = int(rng.integers(4, 7))
        fill = int(rng.integers(170, 256))
        _DRAWERS[int(label_index)](draw, center, size, fill)
        labels[int(label_index)] = 1.0

    if rng.random() < 0.20:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.8))))

    array = np.asarray(image, dtype=np.float32)
    noise = rng.normal(0.0, 8.0, size=array.shape).astype(np.float32)
    array = np.clip(array + noise, 0.0, 255.0) / 255.0
    return array[None, ...], labels


def generate_dataset(num_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    labels = np.zeros((num_samples, len(LABEL_NAMES)), dtype=np.float32)

    for index in range(num_samples):
        image, label = generate_shape_sample(rng)
        images[index] = image
        labels[index] = label

    return torch.from_numpy(images), torch.from_numpy(labels)


class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images.float()
        self.labels = labels.float()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


@dataclass
class DemoSplits:
    train_images: torch.Tensor
    train_labels: torch.Tensor
    val_images: torch.Tensor
    val_labels: torch.Tensor
    query_images: torch.Tensor
    query_labels: torch.Tensor
    eval_images: torch.Tensor
    eval_labels: torch.Tensor


def select_small_multilabel_subset(
    labels: torch.Tensor,
    *,
    subset_size: int,
    seed: int,
    min_examples_per_label: int = 4,
) -> torch.Tensor:
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")

    labels_np = labels.cpu().numpy().astype(np.float32)
    num_samples, num_labels = labels_np.shape
    if subset_size > num_samples:
        raise ValueError("subset_size cannot exceed the number of available samples.")

    rng = np.random.default_rng(seed)
    candidate_order = rng.permutation(num_samples)
    chosen: list[int] = []
    chosen_set: set[int] = set()
    per_label_counts = np.zeros(num_labels, dtype=np.int32)
    target = np.full(num_labels, min_examples_per_label, dtype=np.int32)

    for index in candidate_order:
        sample_labels = labels_np[index] > 0.5
        if not sample_labels.any():
            continue
        if np.any(per_label_counts[sample_labels] < target[sample_labels]):
            chosen.append(int(index))
            chosen_set.add(int(index))
            per_label_counts += sample_labels.astype(np.int32)
            if len(chosen) == subset_size:
                return torch.tensor(chosen, dtype=torch.long)
            if np.all(per_label_counts >= target):
                break

    for index in candidate_order:
        if len(chosen) == subset_size:
            break
        if int(index) in chosen_set:
            continue
        chosen.append(int(index))
        chosen_set.add(int(index))

    return torch.tensor(chosen[:subset_size], dtype=torch.long)


def build_demo_splits(
    *,
    seed: int = 42,
    train_size: int = 5000,
    val_size: int = 1000,
    query_size: int = 1200,
    eval_size: int = 1200,
) -> DemoSplits:
    train_images, train_labels = generate_dataset(train_size, seed=seed)
    val_images, val_labels = generate_dataset(val_size, seed=seed + 1)
    query_images, query_labels = generate_dataset(query_size, seed=seed + 2)
    eval_images, eval_labels = generate_dataset(eval_size, seed=seed + 3)

    return DemoSplits(
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        query_images=query_images,
        query_labels=query_labels,
        eval_images=eval_images,
        eval_labels=eval_labels,
    )
