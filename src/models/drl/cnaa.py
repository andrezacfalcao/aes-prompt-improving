"""
CNAA (Contrastive Norm-Angular Alignment)

Disentangles quality vs content information in EQ-net features via:
1. Data Augmentation: Creates high/low quality and cross-prompt pairs
2. NIA (Norm-Invariant Alignment): Aligns norms for same quality
3. ASA (Angular-Shift Alignment): Aligns angles for same content (prompt)

Implementação baseada no artigo ACL 2023.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CNAAConfig:
    """Configuration for CNAA pre-training."""

    delta_h: float = 0.8
    delta_l: float = 0.3

    margin_m1: float = 0.1
    margin_m2: float = 0.1

    score_reduction_mu: float = 0.4
    score_reduction_sigma: float = 1.0

    augmentation_ratio: float = 1.0

    nia_weight: float = 1.0
    asa_weight: float = 1.0

    pretrain_epochs: int = 5
    pretrain_learning_rate: float = 5e-5


class DataAugmenter:
    """
    Creates augmented data for CNAA pre-training.

    Generates:
    - D_o: Original data with quality and content labels
    - D_d: Derived data via text concatenation
    """

    def __init__(self, config: CNAAConfig):
        self.config = config

    def filter_by_quality(
        self,
        essays: List[str],
        scores: List[float],
        prompts: List[int]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter essays into high and low quality groups.

        Args:
            essays: List of essay texts
            scores: List of normalized scores [0, 1]
            prompts: List of prompt IDs

        Returns:
            high_quality: List of {essay, score, prompt, quality=1}
            low_quality: List of {essay, score, prompt, quality=0}
        """
        high_quality = []
        low_quality = []

        for essay, score, prompt in zip(essays, scores, prompts):
            if score >= self.config.delta_h:
                high_quality.append({
                    'essay': essay,
                    'score': score,
                    'prompt': prompt,
                    'quality': 1,
                    'content': prompt
                })
            elif score <= self.config.delta_l:
                low_quality.append({
                    'essay': essay,
                    'score': score,
                    'prompt': prompt,
                    'quality': 0,
                    'content': prompt
                })

        return high_quality, low_quality

    def create_derived_data(
        self,
        original_data: List[Dict]
    ) -> List[Dict]:
        """
        Create derived data via text concatenation.

        Generates 4 types:
        1. e_i ⊕ e_j (same quality, different prompt)
        2. e_i ⊕ e_j (same quality, same prompt)
        3. e_i ⊕ e_j (different quality)
        4. e_i ⊕ p_j (essay + random prompt text)

        Args:
            original_data: List of original samples

        Returns:
            derived_data: List of augmented samples
        """
        derived_data = []
        n_samples = int(len(original_data) * self.config.augmentation_ratio)

        for _ in range(n_samples):
            i, j = random.sample(range(len(original_data)), 2)
            sample_i = original_data[i]
            sample_j = original_data[j]

            augmented_essay = sample_i['essay'] + ' ' + sample_j['essay']

            score_reduction = np.random.normal(
                self.config.score_reduction_mu,
                self.config.score_reduction_sigma
            )
            derived_score = max(sample_i['score'], sample_j['score']) - score_reduction
            derived_score = np.clip(derived_score, 0, 1)

            derived_content = random.choice([sample_i['content'], sample_j['content']])

            if derived_score >= self.config.delta_h:
                derived_quality = 1
            elif derived_score <= self.config.delta_l:
                derived_quality = 0
            else:
                continue

            derived_data.append({
                'essay': augmented_essay,
                'score': derived_score,
                'prompt': derived_content,
                'quality': derived_quality,
                'content': derived_content,
                'is_derived': True
            })

        return derived_data

    def augment_dataset(
        self,
        essays: List[str],
        scores: List[float],
        prompts: List[int]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Complete augmentation pipeline.

        Args:
            essays: Original essay texts
            scores: Normalized scores
            prompts: Prompt IDs

        Returns:
            original_data: Filtered original data with labels
            derived_data: Augmented data
        """
        high_quality, low_quality = self.filter_by_quality(essays, scores, prompts)

        original_data = high_quality + low_quality

        derived_data = self.create_derived_data(original_data)

        return original_data, derived_data


class CNAALoss(nn.Module):
    """
    CNAA Loss: Norm-Invariant Alignment + Angular-Shift Alignment

    L_CNAA = L_NIA + L_ASA
    """

    def __init__(self, config: CNAAConfig):
        super().__init__()
        self.config = config

    def norm_invariant_alignment(
        self,
        features: torch.Tensor,
        quality_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        NIA Loss: Aligns feature norms based on quality.

        L_NIA = {
            ||v_i|| - ||v_j||,                    if q_i = q_j
            max(0, m_1 - ||v_i|| - ||v_j||),      if q_i ≠ q_j
        }

        Args:
            features: [batch_size, hidden_dim] - EQ-net features
            quality_labels: [batch_size] - quality labels {0, 1}

        Returns:
            loss: NIA loss
        """
        batch_size = features.size(0)

        norms = torch.norm(features, p=2, dim=1)

        quality_expanded_i = quality_labels.unsqueeze(1)
        quality_expanded_j = quality_labels.unsqueeze(0)
        same_quality = (quality_expanded_i == quality_expanded_j).float()

        norm_expanded_i = norms.unsqueeze(1)
        norm_expanded_j = norms.unsqueeze(0)
        norm_diff = torch.abs(norm_expanded_i - norm_expanded_j)

        loss_same = same_quality * norm_diff

        different_quality = 1 - same_quality
        loss_diff = different_quality * torch.clamp(
            self.config.margin_m1 - norm_diff,
            min=0
        )

        mask = 1 - torch.eye(batch_size, device=features.device)
        loss = (loss_same + loss_diff) * mask
        loss = loss.sum() / (mask.sum() + 1e-8)

        return loss

    def angular_shift_alignment(
        self,
        features: torch.Tensor,
        content_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        ASA Loss: Aligns feature angles based on content (prompt).

        L_ASA = {
            1 - cos(v_i, v_j),                    if c_i = c_j
            max(0, cos(v_i, v_j) - m_2),          if c_i ≠ c_j
        }

        Args:
            features: [batch_size, hidden_dim] - EQ-net features
            content_labels: [batch_size] - content/prompt labels

        Returns:
            loss: ASA loss
        """
        batch_size = features.size(0)

        features_norm = F.normalize(features, p=2, dim=1)

        cosine_sim = torch.mm(features_norm, features_norm.t())

        content_expanded_i = content_labels.unsqueeze(1)
        content_expanded_j = content_labels.unsqueeze(0)
        same_content = (content_expanded_i == content_expanded_j).float()

        loss_same = same_content * (1 - cosine_sim)

        different_content = 1 - same_content
        loss_diff = different_content * torch.clamp(
            cosine_sim - self.config.margin_m2,
            min=0
        )

        mask = 1 - torch.eye(batch_size, device=features.device)
        loss = (loss_same + loss_diff) * mask
        loss = loss.sum() / (mask.sum() + 1e-8)

        return loss

    def forward(
        self,
        features: torch.Tensor,
        quality_labels: torch.Tensor,
        content_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total CNAA loss.

        Args:
            features: [batch_size, hidden_dim] - EQ-net features
            quality_labels: [batch_size] - quality labels
            content_labels: [batch_size] - content/prompt labels

        Returns:
            Dict with 'loss', 'nia_loss', 'asa_loss'
        """
        nia_loss = self.norm_invariant_alignment(features, quality_labels)
        asa_loss = self.angular_shift_alignment(features, content_labels)

        total_loss = (
            self.config.nia_weight * nia_loss +
            self.config.asa_weight * asa_loss
        )

        return {
            'loss': total_loss,
            'nia_loss': nia_loss.item(),
            'asa_loss': asa_loss.item()
        }
