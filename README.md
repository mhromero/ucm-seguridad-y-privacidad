# Privacy and Security

## Overview

This repository is a collection of coursework for Privacy and Security. It brings together executed notebooks, a Unix-permissions exercise, and model-extraction materials.

## Tech Stack

Python, Jupyter Notebook, PyTorch, torchvision, NumPy, matplotlib, Transformers, TextAttack, and Bash.

## Getting Started

### Requirements

No runtime setup is required to read the collection. The notebooks are preserved with their executed outputs and results.

### Installation

Installation is not required for the intended read-only use of this repository.

### Running

The repository is not presented as a reproducible end-to-end environment. Some notebooks refer to data or images used in their original execution that are not included here. The multilabel model-extraction implementation retains its own historical setup and run instructions in its [README](practicas/robo-modelo/implementacion-multietiqueta/README.md).

## Contents

- [CIFAR-10 poisoning](practicas/envenenamiento-cifar10/notebook.ipynb): image-classification poisoning attacks.
- [Text poisoning](practicas/envenenamiento-texto/notebook.ipynb): a sentiment-analysis attack using Sentiment140 and TextAttack.
- [Nightshade](practicas/nightshade/notebook.ipynb): an adversarial image-perturbation demonstration.
- [Unix permissions](practicas/permisos-unix/setup-arbol.sh): ownership and group configuration for a directory tree.
- [Model extraction](practicas/robo-modelo/robo_modelo.ipynb): [victim-model notebook](practicas/robo-modelo/modelo-victima.ipynb), extraction notebook and report, practice artifacts, and a [multilabel implementation](practicas/robo-modelo/implementacion-multietiqueta/README.md).

## Academic Context

Materials for the Privacy and Security course at Universidad Complutense de Madrid.
