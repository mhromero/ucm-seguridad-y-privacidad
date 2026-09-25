from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from bundle_io import save_bundle
from data import LABEL_NAMES, build_demo_splits
from models import load_own_teacher_checkpoint, load_serialized_module_checkpoint
from peer_model_adapter import load_peer_model
from utils import get_device, predict_probabilities, set_seed


def load_queryable_model(
    *,
    checkpoint_path: str | Path,
    checkpoint_kind: str,
    device: torch.device,
) -> tuple[nn.Module, list[str]]:
    if checkpoint_kind == "own_teacher":
        model, checkpoint = load_own_teacher_checkpoint(checkpoint_path, device=device)
        return model, checkpoint.get("label_names", LABEL_NAMES)

    if checkpoint_kind == "serialized_module":
        model = load_serialized_module_checkpoint(checkpoint_path, device=device)
        return model, LABEL_NAMES

    if checkpoint_kind == "adapter":
        model = load_peer_model(checkpoint_path, device=device)
        model = model.to(device)
        model.eval()
        return model, LABEL_NAMES

    raise ValueError(f"Unknown checkpoint kind: {checkpoint_kind}")


def export_model_bundles(
    *,
    checkpoint_path: str | Path,
    checkpoint_kind: str = "own_teacher",
    query_bundle_path: str | Path = "artifacts/query_bundle.pt",
    eval_bundle_path: str | Path = "artifacts/eval_bundle.pt",
    seed: int = 42,
    query_size: int = 1200,
    eval_size: int = 1200,
    batch_size: int = 256,
    source_name: str | None = None,
) -> tuple[Path, Path]:
    set_seed(seed)
    device = get_device()

    model, label_names = load_queryable_model(
        checkpoint_path=checkpoint_path,
        checkpoint_kind=checkpoint_kind,
        device=device,
    )

    splits = build_demo_splits(
        seed=seed,
        query_size=query_size,
        eval_size=eval_size,
    )

    query_probs = predict_probabilities(model, splits.query_images, device=device, batch_size=batch_size)
    eval_probs = predict_probabilities(model, splits.eval_images, device=device, batch_size=batch_size)
    source_name = source_name or checkpoint_kind

    query_bundle_path = save_bundle(
        query_bundle_path,
        images=splits.query_images,
        teacher_probs=query_probs,
        labels=splits.query_labels,
        label_names=label_names,
        metadata={
            "source": source_name,
            "seed": seed,
            "purpose": "student_training",
            "checkpoint_kind": checkpoint_kind,
        },
    )
    eval_bundle_path = save_bundle(
        eval_bundle_path,
        images=splits.eval_images,
        teacher_probs=eval_probs,
        labels=splits.eval_labels,
        label_names=label_names,
        metadata={
            "source": source_name,
            "seed": seed,
            "purpose": "final_evaluation",
            "checkpoint_kind": checkpoint_kind,
        },
    )

    print(f"Saved query bundle to {query_bundle_path}")
    print(f"Saved evaluation bundle to {eval_bundle_path}")
    return query_bundle_path, eval_bundle_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a teacher-like checkpoint and export distillation bundles.")
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/teacher_model.pt")
    parser.add_argument(
        "--checkpoint-kind",
        choices=["own_teacher", "serialized_module", "adapter"],
        default="own_teacher",
    )
    parser.add_argument("--query-bundle", type=str, default="artifacts/query_bundle.pt")
    parser.add_argument("--eval-bundle", type=str, default="artifacts/eval_bundle.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-size", type=int, default=1200)
    parser.add_argument("--eval-size", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--source-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model_bundles(
        checkpoint_path=args.checkpoint_path,
        checkpoint_kind=args.checkpoint_kind,
        query_bundle_path=args.query_bundle,
        eval_bundle_path=args.eval_bundle,
        seed=args.seed,
        query_size=args.query_size,
        eval_size=args.eval_size,
        batch_size=args.batch_size,
        source_name=args.source_name,
    )
