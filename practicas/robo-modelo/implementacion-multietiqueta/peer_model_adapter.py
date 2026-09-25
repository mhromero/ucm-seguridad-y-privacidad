from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def load_peer_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> nn.Module:
    """
    Edit this function when a classmate sends a state_dict checkpoint plus their architecture.

    Expected return:
    - a torch.nn.Module
    - moved to `device`
    - in eval mode

    If they only send a raw .pt with no architecture information, you usually cannot recover
    the model reliably. In that case ask for either:
    - the full serialized module, or
    - the model class / source code needed to instantiate it.
    """

    raise NotImplementedError(
        "Implement peer_model_adapter.load_peer_model() with your classmate's architecture "
        "when they send a state_dict-based checkpoint."
    )

