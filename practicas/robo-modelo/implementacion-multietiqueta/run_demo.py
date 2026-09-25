from __future__ import annotations

import argparse

from extract_model import run_extraction
from query_model import export_model_bundles
from train_teacher import train_teacher_model
from visualizations import generate_all_visualizations


def run_demo(
    *,
    seed: int = 42,
    teacher_epochs: int = 8,
    extraction_epochs: int = 8,
    batch_size: int = 64,
    query_size: int = 1200,
    eval_size: int = 1200,
    search_preset: str = "standard",
) -> None:
    train_teacher_model(
        seed=seed,
        epochs=teacher_epochs,
        batch_size=batch_size,
    )
    export_model_bundles(
        checkpoint_path="artifacts/teacher_model.pt",
        checkpoint_kind="own_teacher",
        seed=seed,
        query_size=query_size,
        eval_size=eval_size,
    )
    run_extraction(
        search_preset=search_preset,
        seed=seed,
        epochs=extraction_epochs,
        batch_size=batch_size,
    )
    generate_all_visualizations()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full multilabel extraction demo pipeline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-epochs", type=int, default=8)
    parser.add_argument("--extraction-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-size", type=int, default=1200)
    parser.add_argument("--eval-size", type=int, default=1200)
    parser.add_argument("--search-preset", choices=["small", "standard"], default="standard")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        seed=args.seed,
        teacher_epochs=args.teacher_epochs,
        extraction_epochs=args.extraction_epochs,
        batch_size=args.batch_size,
        query_size=args.query_size,
        eval_size=args.eval_size,
        search_preset=args.search_preset,
    )
