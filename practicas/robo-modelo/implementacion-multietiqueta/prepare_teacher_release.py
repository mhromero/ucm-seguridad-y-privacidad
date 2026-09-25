from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bundle_io import save_bundle
from data import LABEL_NAMES, build_demo_splits, select_small_multilabel_subset
from train_teacher import train_teacher_model
from utils import ensure_dir, get_device, predict_probabilities


def _write_student_instructions(path: Path, *, public_bundle_name: str, label_names: list[str]) -> None:
    instructions = {
        "task": "Replicar el comportamiento de un modelo multietiqueta a partir de sus salidas.",
        "public_bundle": public_bundle_name,
        "tiny_train_subset_bundle": "tiny_train_subset.pt",
        "image_shape": [1, 32, 32],
        "label_names": label_names,
        "notes": [
            "El bundle publico contiene entradas y salidas del teacher.",
            "No incluye etiquetas reales del dataset.",
            "Tu modelo student debe imitar las probabilidades del teacher.",
            "Tambien se incluye un subconjunto muy pequeno del train split como referencia.",
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(instructions, handle, indent=2)


def prepare_teacher_release(
    *,
    output_dir: str | Path = "artifacts/teacher_release",
    checkpoint_path: str | Path = "artifacts/teacher_model.pt",
    seed: int = 42,
    teacher_epochs: int = 8,
    batch_size: int = 64,
    train_size: int = 5000,
    val_size: int = 1000,
    public_query_size: int = 1200,
    private_eval_size: int = 1200,
    public_train_subset_size: int = 24,
    include_public_checkpoint: bool = False,
) -> dict[str, str]:
    model, metrics = train_teacher_model(
        seed=seed,
        epochs=teacher_epochs,
        batch_size=batch_size,
        train_size=train_size,
        val_size=val_size,
        checkpoint_path=checkpoint_path,
    )

    device = get_device()
    model = model.to(device)
    model.eval()

    splits = build_demo_splits(
        seed=seed,
        query_size=public_query_size,
        eval_size=private_eval_size,
    )

    public_teacher_probs = predict_probabilities(
        model,
        splits.query_images,
        device=device,
        batch_size=batch_size,
    )
    private_teacher_probs = predict_probabilities(
        model,
        splits.eval_images,
        device=device,
        batch_size=batch_size,
    )
    public_train_subset_indices = select_small_multilabel_subset(
        splits.train_labels,
        subset_size=public_train_subset_size,
        seed=seed + 101,
        min_examples_per_label=4,
    )
    public_train_subset_images = splits.train_images[public_train_subset_indices]
    public_train_subset_labels = splits.train_labels[public_train_subset_indices]
    public_train_subset_teacher_probs = predict_probabilities(
        model,
        public_train_subset_images,
        device=device,
        batch_size=batch_size,
    )

    output_dir = Path(output_dir)
    public_dir = ensure_dir(output_dir / "public")
    private_dir = ensure_dir(output_dir / "private")

    public_bundle = save_bundle(
        public_dir / "challenge_bundle.pt",
        images=splits.query_images,
        teacher_probs=public_teacher_probs,
        label_names=LABEL_NAMES,
        metadata={
            "split": "public_challenge",
            "seed": seed,
            "teacher_epochs": teacher_epochs,
            "description": "Bundle compartible con entradas y salidas del teacher.",
        },
    )
    public_train_subset_bundle = save_bundle(
        public_dir / "tiny_train_subset.pt",
        images=public_train_subset_images,
        teacher_probs=public_train_subset_teacher_probs,
        labels=public_train_subset_labels,
        label_names=LABEL_NAMES,
        metadata={
            "split": "public_tiny_train_subset",
            "seed": seed,
            "train_size": train_size,
            "selected_indices": public_train_subset_indices.tolist(),
            "description": "Subconjunto pequeno del train split usado para entrenar el teacher.",
        },
    )
    private_eval_bundle = save_bundle(
        private_dir / "eval_bundle_private.pt",
        images=splits.eval_images,
        teacher_probs=private_teacher_probs,
        labels=splits.eval_labels,
        label_names=LABEL_NAMES,
        metadata={
            "split": "private_evaluation",
            "seed": seed,
            "teacher_epochs": teacher_epochs,
            "description": "Bundle privado para evaluar students enviados por compañeros.",
        },
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    private_checkpoint_path = private_dir / "teacher_model.pt"
    torch.save(checkpoint, private_checkpoint_path)

    if include_public_checkpoint:
        torch.save(checkpoint, public_dir / "teacher_model.pt")

    _write_student_instructions(
        public_dir / "challenge_manifest.json",
        public_bundle_name=public_bundle.name,
        label_names=LABEL_NAMES,
    )

    summary_path = output_dir / "release_summary.json"
    summary = {
        "public_bundle": str(public_bundle),
        "public_tiny_train_subset": str(public_train_subset_bundle),
        "private_eval_bundle": str(private_eval_bundle),
        "private_teacher_checkpoint": str(private_checkpoint_path),
        "include_public_checkpoint": include_public_checkpoint,
        "teacher_architecture": "TeacherCNN",
        "teacher_validation_metrics": metrics,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTeacher release prepared")
    print("=" * 48)
    print(f"Public bundle:            {public_bundle}")
    print(f"Tiny train subset:        {public_train_subset_bundle}")
    print(f"Private eval bundle:      {private_eval_bundle}")
    print(f"Private teacher weights:  {private_checkpoint_path}")
    if include_public_checkpoint:
        print(f"Public teacher weights:   {public_dir / 'teacher_model.pt'}")
    print("=" * 48)

    return {
        "public_bundle": str(public_bundle),
        "public_tiny_train_subset": str(public_train_subset_bundle),
        "private_eval_bundle": str(private_eval_bundle),
        "private_teacher_checkpoint": str(private_checkpoint_path),
        "summary": str(summary_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y empaqueta el teacher para compartir la práctica.")
    parser.add_argument("--output-dir", type=str, default="artifacts/teacher_release")
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/teacher_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--public-query-size", type=int, default=1200)
    parser.add_argument("--private-eval-size", type=int, default=1200)
    parser.add_argument("--public-train-subset-size", type=int, default=24)
    parser.add_argument("--include-public-checkpoint", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_teacher_release(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        teacher_epochs=args.teacher_epochs,
        batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
        public_query_size=args.public_query_size,
        private_eval_size=args.private_eval_size,
        public_train_subset_size=args.public_train_subset_size,
        include_public_checkpoint=args.include_public_checkpoint,
    )
