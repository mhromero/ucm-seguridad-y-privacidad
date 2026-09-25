from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import LABEL_NAMES, TensorImageDataset, build_demo_splits
from metrics import multilabel_metrics
from models import NUM_LABELS, TeacherCNN
from utils import get_device, predict_probabilities, set_seed


def train_teacher_model(
    *,
    seed: int = 42,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    train_size: int = 5000,
    val_size: int = 1000,
    checkpoint_path: str | Path = "artifacts/teacher_model.pt",
) -> tuple[TeacherCNN, dict[str, float]]:
    set_seed(seed)
    device = get_device()
    print(f"Training teacher on: {device}")

    splits = build_demo_splits(
        seed=seed,
        train_size=train_size,
        val_size=val_size,
    )
    train_loader = DataLoader(
        TensorImageDataset(splits.train_images, splits.train_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    model = TeacherCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_state = None
    best_micro_f1 = -1.0
    best_metrics: dict[str, float] = {}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_probs = predict_probabilities(model, splits.val_images, device=device)
        val_metrics = multilabel_metrics(
            splits.val_labels.numpy(),
            val_probs.numpy(),
        )
        average_loss = running_loss / max(1, len(train_loader))
        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {average_loss:.4f} | "
            f"Val subset acc: {val_metrics['subset_accuracy']:.3f} | "
            f"Val micro-F1: {val_metrics['micro_f1']:.3f}"
        )

        if val_metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = val_metrics["micro_f1"]
            best_metrics = val_metrics
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Teacher training finished without a valid checkpoint.")

    model.load_state_dict(best_state)

    checkpoint = {
        "state_dict": model.state_dict(),
        "architecture": "TeacherCNN",
        "num_labels": NUM_LABELS,
        "label_names": LABEL_NAMES,
        "seed": seed,
        "train_size": train_size,
        "val_size": val_size,
        "metrics": best_metrics,
    }
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    print(
        "\nTeacher model saved. "
        f"Best validation subset accuracy: {best_metrics['subset_accuracy']:.3f} | "
        f"micro-F1: {best_metrics['micro_f1']:.3f}"
    )
    return model, best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the demo teacher model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/teacher_model.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_teacher_model(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_size=args.train_size,
        val_size=args.val_size,
        checkpoint_path=args.checkpoint_path,
    )
