from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from bundle_io import load_bundle
from metrics import build_report, teacher_student_metrics
from models import NUM_LABELS, build_student_from_config
from utils import get_device, predict_probabilities, set_seed


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    clipped_teacher = teacher_probabilities.clamp(1e-4, 1.0 - 1e-4)
    softened_targets = torch.sigmoid(torch.logit(clipped_teacher) / temperature)
    scaled_logits = student_logits / temperature
    return F.binary_cross_entropy_with_logits(scaled_logits, softened_targets) * (temperature**2)


def search_space_from_preset(preset: str) -> list[dict[str, Any]]:
    if preset == "small":
        return [
            {"conv_channels": (16, 32), "classifier_hidden": 64, "dropout": 0.10, "learning_rate": 1e-3},
            {"conv_channels": (16, 32), "classifier_hidden": 128, "dropout": 0.25, "learning_rate": 1e-3},
            {"conv_channels": (24, 48), "classifier_hidden": 128, "dropout": 0.10, "learning_rate": 5e-4},
            {"conv_channels": (16, 32, 64), "classifier_hidden": 128, "dropout": 0.25, "learning_rate": 5e-4},
        ]
    if preset == "standard":
        conv_options = [(16, 32), (24, 48), (16, 32, 64)]
        hidden_options = [64, 128]
        dropout_options = [0.10, 0.25]
        return [
            {
                "conv_channels": conv_channels,
                "classifier_hidden": classifier_hidden,
                "dropout": dropout,
                "learning_rate": 1e-3,
            }
            for conv_channels, classifier_hidden, dropout in itertools.product(
                conv_options,
                hidden_options,
                dropout_options,
            )
        ]
    raise ValueError(f"Unknown preset: {preset}")


def _config_to_name(config: dict[str, Any]) -> str:
    channel_text = "-".join(str(channel) for channel in config["conv_channels"])
    return (
        f"conv[{channel_text}]_hidden{config['classifier_hidden']}_"
        f"drop{config['dropout']}_lr{config['learning_rate']}"
    )


def _make_loaders(
    images: torch.Tensor,
    teacher_probs: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    val_fraction: float,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(images), generator=generator)
    val_size = max(1, int(len(images) * val_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = TensorDataset(images[train_indices], teacher_probs[train_indices])
    val_dataset = TensorDataset(images[val_indices], teacher_probs[val_indices])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    )


def _train_one_config(
    *,
    config: dict[str, Any],
    train_loader: DataLoader,
    val_images: torch.Tensor,
    val_teacher_probs: torch.Tensor,
    epochs: int,
    temperature: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    student = build_student_from_config({**config, "num_labels": NUM_LABELS}).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(config["learning_rate"]))

    best_state = None
    best_score = -1.0
    best_mae = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0

        for images, teacher_probs in train_loader:
            images = images.to(device)
            teacher_probs = teacher_probs.to(device)

            optimizer.zero_grad()
            logits = student(images)
            loss = distillation_loss(logits, teacher_probs, temperature=temperature)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_student_probs = predict_probabilities(student, val_images, device=device)
        val_metrics = teacher_student_metrics(
            val_teacher_probs.numpy(),
            val_student_probs.numpy(),
        )
        average_loss = running_loss / max(1, len(train_loader))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": average_loss,
                "validation_exact_agreement": val_metrics["exact_agreement"],
                "validation_probability_mae": val_metrics["probability_mae"],
            }
        )

        score = float(val_metrics["exact_agreement"])
        mae = float(val_metrics["probability_mae"])
        if score > best_score or (score == best_score and mae < best_mae):
            best_score = score
            best_mae = mae
            best_state = {key: value.cpu() for key, value in student.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"Config {_config_to_name(config)} did not produce a valid checkpoint.")

    return {
        "config": config,
        "config_name": _config_to_name(config),
        "best_state": best_state,
        "best_validation_exact_agreement": best_score,
        "best_validation_probability_mae": best_mae,
        "history": history,
    }


def _train_final_student(
    *,
    config: dict[str, Any],
    images: torch.Tensor,
    teacher_probs: torch.Tensor,
    epochs: int,
    temperature: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    set_seed(seed)
    student = build_student_from_config({**config, "num_labels": NUM_LABELS}).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(config["learning_rate"]))
    loader = DataLoader(TensorDataset(images, teacher_probs), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        student.train()
        for batch_images, batch_teacher_probs in loader:
            batch_images = batch_images.to(device)
            batch_teacher_probs = batch_teacher_probs.to(device)

            optimizer.zero_grad()
            logits = student(batch_images)
            loss = distillation_loss(logits, batch_teacher_probs, temperature=temperature)
            loss.backward()
            optimizer.step()

    return student


def run_extraction(
    *,
    query_bundle_path: str | Path = "artifacts/query_bundle.pt",
    eval_bundle_path: str | Path = "artifacts/eval_bundle.pt",
    output_checkpoint_path: str | Path = "artifacts/best_student.pt",
    output_report_path: str | Path = "artifacts/extraction_report.json",
    search_preset: str = "standard",
    seed: int = 42,
    epochs: int = 8,
    batch_size: int = 64,
    temperature: float = 2.0,
    val_fraction: float = 0.15,
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device()
    print(f"Extracting student on: {device}")

    query_bundle = load_bundle(query_bundle_path)
    eval_bundle = load_bundle(eval_bundle_path)

    if "teacher_probs" not in query_bundle:
        raise ValueError("The query bundle must contain 'teacher_probs'.")

    images = query_bundle["images"].float()
    teacher_probs = query_bundle["teacher_probs"].float()
    label_names = query_bundle.get("label_names")

    train_loader, val_loader = _make_loaders(
        images,
        teacher_probs,
        batch_size=batch_size,
        seed=seed,
        val_fraction=val_fraction,
    )
    val_images, val_teacher_probs = val_loader.dataset.tensors

    results: list[dict[str, Any]] = []
    for config in search_space_from_preset(search_preset):
        print(f"\n[Grid search] { _config_to_name(config) }")
        result = _train_one_config(
            config=config,
            train_loader=train_loader,
            val_images=val_images,
            val_teacher_probs=val_teacher_probs,
            epochs=epochs,
            temperature=temperature,
            seed=seed,
            device=device,
        )
        results.append(result)
        print(
            f"Best validation exact agreement: {result['best_validation_exact_agreement']:.3f} | "
            f"probability MAE: {result['best_validation_probability_mae']:.4f}"
        )

    best_result = sorted(
        results,
        key=lambda item: (
            item["best_validation_exact_agreement"],
            -item["best_validation_probability_mae"],
        ),
        reverse=True,
    )[0]
    best_config = best_result["config"]

    print(f"\nBest config: {best_result['config_name']}")
    student = _train_final_student(
        config=best_config,
        images=images,
        teacher_probs=teacher_probs,
        epochs=epochs,
        temperature=temperature,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    eval_images = eval_bundle["images"].float()
    eval_teacher_probs = eval_bundle["teacher_probs"].float()
    eval_true_labels = eval_bundle.get("labels")
    student_probs = predict_probabilities(student, eval_images, device=device)

    report = build_report(
        teacher_probabilities=eval_teacher_probs.numpy(),
        student_probabilities=student_probs.numpy(),
        true_labels=eval_true_labels.numpy() if eval_true_labels is not None else None,
        threshold=0.5,
        label_names=label_names,
    )
    report["search_results"] = [
        {
            "config_name": item["config_name"],
            "config": {
                "conv_channels": list(item["config"]["conv_channels"]),
                "classifier_hidden": item["config"]["classifier_hidden"],
                "dropout": item["config"]["dropout"],
                "learning_rate": item["config"]["learning_rate"],
            },
            "best_validation_exact_agreement": item["best_validation_exact_agreement"],
            "best_validation_probability_mae": item["best_validation_probability_mae"],
        }
        for item in results
    ]
    report["best_config"] = {
        "conv_channels": list(best_config["conv_channels"]),
        "classifier_hidden": best_config["classifier_hidden"],
        "dropout": best_config["dropout"],
        "learning_rate": best_config["learning_rate"],
    }

    output_checkpoint_path = Path(output_checkpoint_path)
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": student.state_dict(),
            "config": report["best_config"],
            "label_names": label_names,
            "report": report,
        },
        output_checkpoint_path,
    )

    output_report_path = Path(output_report_path)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    tvs = report["teacher_vs_student"]
    print("\nExtraction results")
    print("=" * 48)
    print(f"Exact agreement:      {tvs['exact_agreement']:.3f}")
    print(f"Labelwise agreement:  {tvs['labelwise_agreement']:.3f}")
    print(f"Probability MAE:      {tvs['probability_mae']:.4f}")
    if "teacher_vs_truth" in report:
        print(f"Teacher micro-F1:     {report['teacher_vs_truth']['micro_f1']:.3f}")
        print(f"Student micro-F1:     {report['student_vs_truth']['micro_f1']:.3f}")
        print(f"Teacher subset acc:   {report['teacher_vs_truth']['subset_accuracy']:.3f}")
        print(f"Student subset acc:   {report['student_vs_truth']['subset_accuracy']:.3f}")
    print("=" * 48)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a multilabel student model.")
    parser.add_argument("--query-bundle", type=str, default="artifacts/query_bundle.pt")
    parser.add_argument("--eval-bundle", type=str, default="artifacts/eval_bundle.pt")
    parser.add_argument("--output-checkpoint", type=str, default="artifacts/best_student.pt")
    parser.add_argument("--output-report", type=str, default="artifacts/extraction_report.json")
    parser.add_argument("--search-preset", choices=["small", "standard"], default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extraction(
        query_bundle_path=args.query_bundle,
        eval_bundle_path=args.eval_bundle,
        output_checkpoint_path=args.output_checkpoint,
        output_report_path=args.output_report,
        search_preset=args.search_preset,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        val_fraction=args.val_fraction,
    )
