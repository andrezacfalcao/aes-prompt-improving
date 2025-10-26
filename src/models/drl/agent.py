"""Placeholder for DRL-based AES agent.

arquitetura inicial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DRLConfig:
    agent: str = "ppo"
    hidden_size: int = 256
    gamma: float = 0.99
    learning_rate: float = 3e-5
    rollout_steps: int = 128


class DRLAESAgent:
    """Base class for future DRL AES agent."""

    def __init__(self, config: DRLConfig | None = None) -> None:
        self.config = config or DRLConfig()
        raise NotImplementedError(
            "DRL ainda não foi implementado; mantenha-se no baseline por enquanto."
        )

    def train(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def evaluate(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError


