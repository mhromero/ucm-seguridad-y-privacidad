from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from bundle_io import load_bundle
from metrics import build_report
from models import NUM_LABELS, build_student_from_config
from utils import ensure_dir, format_label_set, get_device, predict_probabilities


def _load_student(student_checkpoint: str | Path) -> tuple[torch.nn.Module, list[str], dict]:
    checkpoint = torch.load(student_checkpoint, map_location="cpu")
    config = checkpoint["config"]
    config["conv_channels"] = tuple(config["conv_channels"])
    student = build_student_from_config({**config, "num_labels": NUM_LABELS})
    student.load_state_dict(checkpoint["state_dict"])
    student.eval()
    return student, checkpoint.get("label_names", []), checkpoint.get("report", {})


def _load_prediction_context(
    *,
    eval_bundle_path: str | Path,
    student_checkpoint_path: str | Path,
) -> tuple[dict, torch.Tensor, dict, list[str]]:
    device = get_device()
    bundle = load_bundle(eval_bundle_path)
    student, label_names, saved_report = _load_student(student_checkpoint_path)
    student = student.to(device)
    student_probs = predict_probabilities(student, bundle["images"].float(), device=device)

    report = build_report(
        teacher_probabilities=bundle["teacher_probs"].numpy(),
        student_probabilities=student_probs.numpy(),
        true_labels=bundle["labels"].numpy() if "labels" in bundle else None,
        threshold=0.5,
        label_names=label_names,
    )
    if saved_report.get("search_results"):
        report["search_results"] = saved_report["search_results"]
        report["best_config"] = saved_report.get("best_config", {})
    return bundle, student_probs, report, label_names


def generate_comparison_grid(
    *,
    eval_bundle_path: str | Path = "artifacts/eval_bundle.pt",
    student_checkpoint_path: str | Path = "artifacts/best_student.pt",
    save_path: str | Path = "static/comparison_grid.png",
    num_samples: int = 8,
) -> None:
    bundle, student_probs, _, label_names = _load_prediction_context(
        eval_bundle_path=eval_bundle_path,
        student_checkpoint_path=student_checkpoint_path,
    )
    teacher_probs = bundle["teacher_probs"].numpy()
    images = bundle["images"].numpy()
    labels = bundle.get("labels")
    rng = np.random.default_rng(42)
    indices = rng.choice(len(images), size=min(num_samples, len(images)), replace=False)

    fig = plt.figure(figsize=(15, max(6, len(indices) * 2.2)))
    grid = GridSpec(len(indices), 4, figure=fig, width_ratios=[1, 1.1, 1.4, 1.4])

    for row, index in enumerate(indices):
        image = images[index, 0]
        teacher_sample = teacher_probs[index]
        student_sample = student_probs[index].numpy()

        ax_image = fig.add_subplot(grid[row, 0])
        ax_image.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        title = "Input sample"
        if labels is not None:
            title = format_label_set(labels[index], label_names)
        ax_image.set_title(title, fontsize=9)
        ax_image.axis("off")

        ax_truth = fig.add_subplot(grid[row, 1])
        teacher_binary = (teacher_sample >= 0.5).astype(int)
        student_binary = (student_sample >= 0.5).astype(int)
        status = "MATCH" if np.array_equal(teacher_binary, student_binary) else "DIFF"
        ax_truth.axis("off")
        ax_truth.text(0.0, 0.80, f"Teacher labels: {format_label_set(teacher_binary, label_names)}", fontsize=9)
        ax_truth.text(0.0, 0.52, f"Student labels: {format_label_set(student_binary, label_names)}", fontsize=9)
        ax_truth.text(
            0.0,
            0.24,
            f"Status: {status}",
            fontsize=10,
            color="#1f8f55" if status == "MATCH" else "#c0392b",
            fontweight="bold",
        )

        ax_teacher = fig.add_subplot(grid[row, 2])
        teacher_colors = ["#2ecc71" if value >= 0.5 else "#bfc9ca" for value in teacher_sample]
        ax_teacher.barh(label_names, teacher_sample, color=teacher_colors)
        ax_teacher.set_xlim(0, 1)
        ax_teacher.set_title("Teacher outputs", fontsize=9)

        ax_student = fig.add_subplot(grid[row, 3])
        student_colors = ["#e74c3c" if value >= 0.5 else "#bfc9ca" for value in student_sample]
        ax_student.barh(label_names, student_sample, color=student_colors)
        ax_student.set_xlim(0, 1)
        ax_student.set_title("Student outputs", fontsize=9)

    plt.suptitle("Model extraction comparison on multilabel shape images", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved comparison grid to {save_path}")


def generate_metric_summary(
    *,
    eval_bundle_path: str | Path = "artifacts/eval_bundle.pt",
    student_checkpoint_path: str | Path = "artifacts/best_student.pt",
    save_path: str | Path = "static/metric_summary.png",
) -> None:
    _, _, report, _ = _load_prediction_context(
        eval_bundle_path=eval_bundle_path,
        student_checkpoint_path=student_checkpoint_path,
    )

    metrics_to_plot = ["subset_accuracy", "micro_f1", "macro_f1", "sample_jaccard", "hamming_accuracy"]
    teacher_metrics = report.get("teacher_vs_truth")
    student_metrics = report.get("student_vs_truth")

    if teacher_metrics is None or student_metrics is None:
        values = [
            report["teacher_vs_student"]["exact_agreement"],
            report["teacher_vs_student"]["labelwise_agreement"],
            1.0 - report["teacher_vs_student"]["probability_mae"],
        ]
        labels = ["Exact agreement", "Labelwise agreement", "1 - prob. MAE"]
        plt.figure(figsize=(8, 4))
        bars = plt.bar(labels, values, color=["#2ecc71", "#3498db", "#f39c12"])
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
        plt.ylim(0, 1.05)
        plt.title("Teacher vs student agreement")
        plt.tight_layout()
    else:
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        teacher_values = [teacher_metrics[key] for key in metrics_to_plot]
        student_values = [student_metrics[key] for key in metrics_to_plot]

        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, teacher_values, width=width, label="Teacher", color="#2ecc71")
        plt.bar(x + width / 2, student_values, width=width, label="Student", color="#e74c3c")
        plt.xticks(x, [key.replace("_", "\n") for key in metrics_to_plot])
        plt.ylim(0, 1.05)
        plt.legend()
        plt.title(
            "Teacher vs student on ground truth\n"
            f"Teacher-student exact agreement: {report['teacher_vs_student']['exact_agreement']:.3f}"
        )
        plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved metric summary to {save_path}")


def generate_label_agreement_chart(
    *,
    eval_bundle_path: str | Path = "artifacts/eval_bundle.pt",
    student_checkpoint_path: str | Path = "artifacts/best_student.pt",
    save_path: str | Path = "static/label_agreement.png",
) -> None:
    _, _, report, _ = _load_prediction_context(
        eval_bundle_path=eval_bundle_path,
        student_checkpoint_path=student_checkpoint_path,
    )
    per_label = report["teacher_vs_student"]["per_label_agreement"]
    labels = list(per_label.keys())
    values = list(per_label.values())

    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(labels, values, color="#3498db")
    plt.axhline(report["teacher_vs_student"]["labelwise_agreement"], color="#2c3e50", linestyle="--")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)
    plt.ylim(0, 1.05)
    plt.ylabel("Agreement")
    plt.title("Per-label agreement between teacher and student")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved label agreement chart to {save_path}")


def generate_search_leaderboard(
    *,
    report_path: str | Path = "artifacts/extraction_report.json",
    save_path: str | Path = "static/search_leaderboard.png",
) -> None:
    with Path(report_path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    results = sorted(
        report["search_results"],
        key=lambda item: item["best_validation_exact_agreement"],
        reverse=True,
    )[:6]

    labels = [item["config_name"] for item in results]
    values = [item["best_validation_exact_agreement"] for item in results]

    plt.figure(figsize=(10, 5))
    plt.barh(labels[::-1], values[::-1], color="#9b59b6")
    plt.xlim(0, 1.0)
    plt.xlabel("Validation exact agreement")
    plt.title("Grid search leaderboard")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved search leaderboard to {save_path}")


def generate_all_visualizations(
    *,
    eval_bundle_path: str | Path = "artifacts/eval_bundle.pt",
    student_checkpoint_path: str | Path = "artifacts/best_student.pt",
    report_path: str | Path = "artifacts/extraction_report.json",
    static_dir: str | Path = "static",
) -> None:
    ensure_dir(static_dir)
    generate_comparison_grid(
        eval_bundle_path=eval_bundle_path,
        student_checkpoint_path=student_checkpoint_path,
        save_path=Path(static_dir) / "comparison_grid.png",
    )
    generate_metric_summary(
        eval_bundle_path=eval_bundle_path,
        student_checkpoint_path=student_checkpoint_path,
        save_path=Path(static_dir) / "metric_summary.png",
    )
    generate_label_agreement_chart(
        eval_bundle_path=eval_bundle_path,
        student_checkpoint_path=student_checkpoint_path,
        save_path=Path(static_dir) / "label_agreement.png",
    )
    generate_search_leaderboard(
        report_path=report_path,
        save_path=Path(static_dir) / "search_leaderboard.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static visualizations for extraction results.")
    parser.add_argument("--eval-bundle", type=str, default="artifacts/eval_bundle.pt")
    parser.add_argument("--student-checkpoint", type=str, default="artifacts/best_student.pt")
    parser.add_argument("--report-path", type=str, default="artifacts/extraction_report.json")
    parser.add_argument("--static-dir", type=str, default="static")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all_visualizations(
        eval_bundle_path=args.eval_bundle,
        student_checkpoint_path=args.student_checkpoint,
        report_path=args.report_path,
        static_dir=args.static_dir,
    )
