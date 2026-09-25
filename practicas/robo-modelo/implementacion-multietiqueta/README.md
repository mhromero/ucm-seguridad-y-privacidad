# Multilabel Model Extraction Demo

## Overview

This implementation is inspired by `praetorian-inc/model-extraction-demo`, adapted to a multilabel setting. It uses a synthetic `32x32` image dataset in which each sample can contain multiple geometric shapes.

The workflow has two phases: first, train and package a `teacher` model; later, train a `student` from a bundle of teacher inputs and outputs. The implementation follows the reference project's conceptual flow—training a teacher, querying it, saving its outputs, training a student from `inputs + outputs`, and comparing both models—while using a new synthetic dataset, multilabel outputs, and a grid search over student architectures.

## Tech Stack

Python, PyTorch, NumPy, matplotlib, scikit-learn, Gradio, and Pillow. The exact package requirements are listed in `requirements.txt`.

## Getting Started

### Requirements

Use Python 3 and the dependencies in `requirements.txt`.

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running

This directory is preserved in a coursework collection whose notebooks are intended for reading. The commands below document the original implementation workflow.

#### Prepare a Teacher Release

```bash
python3 prepare_teacher_release.py
```

This trains the `teacher`, creates a public package to share, and creates a private package for evaluating students. It produces:

- `artifacts/teacher_release/public/challenge_bundle.pt`
- `artifacts/teacher_release/public/tiny_train_subset.pt`
- `artifacts/teacher_release/public/challenge_manifest.json`
- `artifacts/teacher_release/private/eval_bundle_private.pt`
- `artifacts/teacher_release/private/teacher_model.pt`

The public directory is `artifacts/teacher_release/public/`. The private directory is `artifacts/teacher_release/private/`, which is intended for evaluating student submissions.

#### Run Individual Stages

Train the teacher separately:

```bash
python3 train_teacher.py
```

Export query and evaluation bundles manually:

```bash
python3 query_model.py --checkpoint-path artifacts/teacher_model.pt --checkpoint-kind own_teacher
```

Extract a student with the standard grid-search preset:

```bash
python3 extract_model.py --search-preset standard
```

Generate comparisons:

```bash
python3 visualizations.py
```

Launch the interactive application:

```bash
python3 app.py
```

Run the complete pipeline:

```bash
python3 run_demo.py
```

## Teacher Release Package

The public package contains `challenge_bundle.pt`, with teacher inputs and outputs; `tiny_train_subset.pt`, a very small training subset with images, labels, and teacher outputs; and `challenge_manifest.json`, with task metadata.

The private package contains `teacher_model.pt`, the teacher weights, and `eval_bundle_private.pt`, an evaluation set for student submissions.

## Working with a Peer Model

The important interface is the bundle format rather than the victim-model checkpoint. `extract_model.py` expects:

- `images`: a `float32` tensor shaped `[N, 1, 32, 32]`
- `teacher_probs`: a `float32` tensor shaped `[N, L]`
- `labels`: an optional `[N, L]` ground-truth tensor
- `label_names`: a list of label names
- `metadata`: optional metadata

Bundles are saved with `torch.save(...)`. If a peer provides inputs and outputs, package them in this format and run:

```bash
python3 extract_model.py \
  --query-bundle ruta/al/query_bundle.pt \
  --eval-bundle ruta/al/eval_bundle.pt
```

When only one sample set is available, it can serve as both `query_bundle` and `eval_bundle`, although the resulting metrics are less clean.

For a checkpoint supplied by a peer, three routes are available:

1. For a checkpoint produced by this repository:

   ```bash
   python3 query_model.py --checkpoint-path artifacts/teacher_model.pt --checkpoint-kind own_teacher
   ```

2. For a serialized module:

   ```bash
   python3 query_model.py --checkpoint-path ruta/al/model.pt --checkpoint-kind serialized_module
   ```

3. For a `state_dict`, edit [peer_model_adapter.py](peer_model_adapter.py) with the peer architecture, then run:

   ```bash
   python3 query_model.py --checkpoint-path ruta/al/model.pt --checkpoint-kind adapter
   ```

A standalone PyTorch `.pt` file is not always sufficient to reconstruct a model; the available approach depends on how the checkpoint was saved. Hard labels also work when stored as `teacher_probs` values of `0` and `1`, although soft probabilities generally provide a stronger extraction signal.

## Student Search and Metrics

The grid search varies convolutional depth, channel width, hidden-layer size, dropout, and learning rate. The `small` preset is faster for iteration; `standard` is more exhaustive.

The report includes `exact_agreement`, `labelwise_agreement`, `probability_mae`, `micro_f1`, `macro_f1`, `subset_accuracy`, `sample_jaccard`, and `hamming_accuracy`. When ground-truth labels are available, it also compares teacher and student predictions against them.

## Results

The original README records the following approximate short CPU-validation results for the complete demo:

- `exact agreement`: ~0.74
- `labelwise agreement`: ~0.93
- `teacher micro-F1`: ~0.91
- `student micro-F1`: ~0.87

## Academic Context

The implementation takes conceptual inspiration from the Praetorian model-extraction demo: separate teacher training, extraction, visualizations, and a simple application. It is implemented from scratch for a multilabel synthetic-image scenario.
