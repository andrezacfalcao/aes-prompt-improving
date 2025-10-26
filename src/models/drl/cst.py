"""
CST (Counterfactual Self-Training)

Disentangles spurious correlation between quality and prompt adherence via:
1. Counterfactual data generation (token substitution)
2. Pre-score guided self-training
3. Score merging (pre-score + pseudo-score)

Implementação baseada no artigo ACL 2023.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class CSTConfig:
    """Configuration for CST fine-tuning."""

    alpha: float = 0.8

    prompt_replace_ratio_random: float = 0.5
    replace_ratios: List[float] = None
    score_multipliers: List[float] = None

    warmup_epochs: int = 2
    warmup_learning_rate: float = 5e-5

    finetune_epochs: int = 3
    finetune_learning_rate: float = 3e-5

    cf_data_ratio: float = 1.0

    def __post_init__(self):
        if self.replace_ratios is None:
            self.replace_ratios = [0.2, 0.3, 0.5]
        if self.score_multipliers is None:
            self.score_multipliers = [1.1, 1.0, 0.9]


class CounterfactualGenerator:
    """
    Generates counterfactual data via token substitution.

    For each original (p_i, e_i, y_i):
    Creates 3 counterfactual instances with different adherence levels.
    """

    def __init__(self, config: CSTConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def substitute_tokens(
        self,
        text: str,
        ratio: float,
        vocab_size: int = None
    ) -> str:
        """
        Substitute random tokens in text.

        Args:
            text: Original text
            ratio: Ratio of tokens to substitute (e.g., 0.2 = 20%)
            vocab_size: Vocabulary size for random token selection

        Returns:
            Modified text with substituted tokens
        """
        if vocab_size is None:
            vocab_size = len(self.tokenizer)

        tokens = self.tokenizer.tokenize(text)

        n_substitute = int(len(tokens) * ratio)
        if n_substitute == 0:
            return text

        positions = random.sample(range(len(tokens)), n_substitute)

        for pos in positions:
            random_id = random.randint(100, vocab_size - 1)
            random_token = self.tokenizer.convert_ids_to_tokens([random_id])[0]
            tokens[pos] = random_token

        modified_text = self.tokenizer.convert_tokens_to_string(tokens)

        return modified_text

    def generate_counterfactuals(
        self,
        prompt: str,
        essay: str,
        score: float
    ) -> List[Dict]:
        """
        Generate counterfactual instances for a single sample.

        Creates:
        1. (p̃_i, p_e^20_i, e_i, ỹ_e^20_i) with score × 1.1
        2. (p̃_i, p_e^30_i, e_i, ỹ_e^30_i) with score × 1.0
        3. (p̃_i, p_e^50_i, e_i, ỹ_e^50_i) with score × 0.9

        Args:
            prompt: Original prompt text
            essay: Original essay text
            score: Original normalized score [0, 1]

        Returns:
            List of counterfactual instances
        """
        counterfactuals = []

        p_tilde = self.substitute_tokens(
            prompt,
            self.config.prompt_replace_ratio_random
        )

        for ratio, multiplier in zip(
            self.config.replace_ratios,
            self.config.score_multipliers
        ):
            p_e = self.substitute_tokens(prompt, ratio)

            pre_score = score * multiplier
            pre_score = min(max(pre_score, 0), 1)

            counterfactuals.append({
                'prompt': p_tilde,
                'prompt_modified': p_e,
                'essay': essay,
                'pre_score': pre_score,
                'original_score': score,
                'is_counterfactual': True
            })

        return counterfactuals

    def create_counterfactual_dataset(
        self,
        prompts: List[str],
        essays: List[str],
        scores: List[float]
    ) -> List[Dict]:
        """
        Create counterfactual dataset for all samples.

        Args:
            prompts: Original prompt texts
            essays: Original essay texts
            scores: Original normalized scores

        Returns:
            List of counterfactual samples
        """
        all_counterfactuals = []

        n_samples = int(len(essays) * self.config.cf_data_ratio)

        for i in range(min(n_samples, len(essays))):
            counterfactuals = self.generate_counterfactuals(
                prompts[i],
                essays[i],
                scores[i]
            )
            all_counterfactuals.extend(counterfactuals)

        return all_counterfactuals


class PreScoreGuidedTrainer:
    """
    Implements pre-score guided self-training.

    Process:
    1. Warmup: Train PANN on original data
    2. Inference: Predict pseudo-scores for counterfactual data
    3. Merge: Combine pre-scores and pseudo-scores
    4. Fine-tune: Continue training with merged scores
    """

    def __init__(self, config: CSTConfig):
        self.config = config

    def merge_scores(
        self,
        pre_scores: torch.Tensor,
        pseudo_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge pre-scores and pseudo-scores.

        y'_i = α·ỹ_e_i + (1-α)·ŷ_i

        Args:
            pre_scores: Pre-defined scores [batch_size]
            pseudo_scores: Model predictions [batch_size]

        Returns:
            merged_scores: [batch_size]
        """
        alpha = self.config.alpha
        merged = alpha * pre_scores + (1 - alpha) * pseudo_scores

        merged = torch.clamp(merged, 0, 1)

        return merged

    def generate_pseudo_scores(
        self,
        model: nn.Module,
        counterfactual_data: List[Dict],
        tokenizer,
        device: str = 'cuda'
    ) -> List[float]:
        """
        Generate pseudo-scores for counterfactual data using trained PANN.

        Args:
            model: Trained PANN model
            counterfactual_data: List of counterfactual samples
            tokenizer: Tokenizer
            device: Device for inference

        Returns:
            pseudo_scores: List of predicted scores
        """
        model.eval()
        pseudo_scores = []

        with torch.no_grad():
            for sample in counterfactual_data:
                prompt_encoding = tokenizer(
                    sample['prompt_modified'],
                    truncation=True,
                    max_length=128,
                    padding='max_length',
                    return_tensors='pt'
                )
                essay_encoding = tokenizer(
                    sample['essay'],
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )

                prompt_input_ids = prompt_encoding['input_ids'].to(device)
                prompt_attention_mask = prompt_encoding['attention_mask'].to(device)
                essay_input_ids = essay_encoding['input_ids'].to(device)
                essay_attention_mask = essay_encoding['attention_mask'].to(device)

                outputs = model(
                    input_ids=essay_input_ids,
                    attention_mask=essay_attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask
                )

                pseudo_score = outputs['predictions'].item()
                pseudo_scores.append(pseudo_score)

        return pseudo_scores

    def update_counterfactual_scores(
        self,
        counterfactual_data: List[Dict],
        pseudo_scores: List[float]
    ) -> List[Dict]:
        """
        Update counterfactual data with merged scores.

        Args:
            counterfactual_data: Counterfactual samples
            pseudo_scores: Predicted scores

        Returns:
            Updated counterfactual data with 'merged_score'
        """
        updated_data = []

        for sample, pseudo_score in zip(counterfactual_data, pseudo_scores):
            pre_score = sample['pre_score']
            merged_score = (
                self.config.alpha * pre_score +
                (1 - self.config.alpha) * pseudo_score
            )
            merged_score = max(0, min(1, merged_score))

            sample['pseudo_score'] = pseudo_score
            sample['merged_score'] = merged_score

            updated_data.append(sample)

        return updated_data


class CSTLoss(nn.Module):
    """
    Combined loss for CST training.

    During warmup: Only MSE loss on original data
    During fine-tuning: MSE loss on original + counterfactual data
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss.

        Args:
            predictions: Model predictions [batch_size]
            targets: Target scores [batch_size]

        Returns:
            loss: MSE loss
        """
        return self.mse_loss(predictions, targets)
