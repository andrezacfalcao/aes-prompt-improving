"""Placeholder implementation for Prompt-Aware Neural Network (PANN).

arquitetura inicial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn


@dataclass
class PANNConfig:
    bert_model: str = "bert-base-uncased"
    bert_hidden_size: int = 768
    num_kernels: int = 8
    kernel_width: float = 0.1
    embedding_dim: int = 300
    fc_layers: int = 2
    fc_hidden_size: int = 256
    dropout: float = 0.1


class PromptAwareAES(nn.Module):
    """Skeleton PANN model. Raise NotImplementedError for now."""

    def __init__(self, config: PANNConfig | None = None) -> None:
        super().__init__()
        self.config = config or PANNConfig()
        raise NotImplementedError(
            "PANN ainda não foi implementado; use TransformerAES como baseline."
        )

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


