"""
Configuration management
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    dropout: float = 0.1
    hidden_size: Optional[int] = 768


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    #12gb vram
    gradient_accumulation_steps: int = 4
    fp16: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    data_path: str = "data/raw/asap"
    source_prompt: int = 1
    target_prompt: int = 2
    val_split: float = 0.1
    random_seed: int = 42


@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    # Logging
    experiment_name: str = "cross_prompt_aes"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
