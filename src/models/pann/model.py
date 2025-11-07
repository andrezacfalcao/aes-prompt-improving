"""
Prompt-Aware Neural Network (PANN) for Cross-Prompt Automated Essay Scoring

Implementação baseada no artigo:
"Improving Domain Generalization for Prompt-Aware Essay Scoring
via Disentangled Representation Learning" (ACL 2023)

Arquitetura:
- EQ-net: Essay Quality Network (BERT-based, prompt-invariant features)
- PA-net: Prompt Adherence Network (prompt-specific features via kernel pooling)
- ESP: Essay Score Predictor (combines EQ and PA features)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class PANNConfig:
    """Configuration for PANN model."""

    bert_model: str = "bert-base-uncased"
    bert_hidden_size: int = 768
    freeze_bert: bool = False

    num_kernels: int = 8
    kernel_width: float = 0.1

    embedding_dim: int = 768  

    fc_layers: int = 2
    fc_hidden_size: int = 256
    dropout: float = 0.1
    activation: str = "relu"

    max_length: int = 512


class EQNet(nn.Module):
    """
    Essay Quality Network

    Extracts prompt-invariant quality features from essays.
    Based on BERT encoder.
    """

    def __init__(self, config: PANNConfig):
        super().__init__()
        self.config = config

        self.encoder = AutoModel.from_pretrained(config.bert_model)

        if config.freeze_bert:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            v_i: Quality features [batch_size, hidden_size]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        v_i = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        return v_i


class PANet(nn.Module):
    """
    Prompt Adherence Network

    Extracts prompt-specific adherence features via:
    1. PE Matching Matrix (cosine similarity between prompt and essay tokens)
    2. Kernel Pooling (soft term frequency using RBF kernels)
    3. Prompt Attention (weighted aggregation)
    """

    def __init__(self, config: PANNConfig):
        super().__init__()
        self.config = config

        self.encoder = AutoModel.from_pretrained(config.bert_model)

        self.num_kernels = config.num_kernels
        self.kernel_width = config.kernel_width

        self.mu = nn.Parameter(
            torch.linspace(-1, 1, self.num_kernels),
            requires_grad=False
        )

        self.attention_transform = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention_context = nn.Parameter(torch.randn(config.embedding_dim))

        self.pooling_types = ['max', 'min', 'avg']

        self.output_dim = self.num_kernels * len(self.pooling_types)

    def compute_matching_matrix(
        self,
        prompt_embeds: torch.Tensor,
        essay_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        essay_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PE Matching Matrix using cosine similarity.

        Args:
            prompt_embeds: [batch_size, m, embed_dim] - prompt token embeddings
            essay_embeds: [batch_size, n, embed_dim] - essay token embeddings
            prompt_mask: [batch_size, m] - prompt attention mask
            essay_mask: [batch_size, n] - essay attention mask

        Returns:
            M: [batch_size, m, n] - matching matrix
        """
        prompt_norm = F.normalize(prompt_embeds, p=2, dim=-1)  # [batch_size, m, embed_dim]
        essay_norm = F.normalize(essay_embeds, p=2, dim=-1)    # [batch_size, n, embed_dim]

        M = torch.bmm(prompt_norm, essay_norm.transpose(1, 2))  # [batch_size, m, n]

        prompt_mask_expanded = prompt_mask.unsqueeze(2)  # [batch_size, m, 1]
        essay_mask_expanded = essay_mask.unsqueeze(1)
        mask = prompt_mask_expanded * essay_mask_expanded

        M = M * mask

        return M

    def kernel_pooling(self, M: torch.Tensor) -> torch.Tensor:
        """
        Apply RBF kernel pooling to matching matrix.

        For each row M_i of the matching matrix, compute soft-TF:
        φ_k(M_i) = Σ_j exp(-(M_ij - μ_k)²/(2σ_k²))

        Args:
            M: [batch_size, m, n] - matching matrix

        Returns:
            K: [batch_size, m, num_kernels] - kernel pooling features
        """
        batch_size, m, n = M.shape

        M_expanded = M.unsqueeze(-1)  # [batch_size, m, n, 1]
        mu_expanded = self.mu.view(1, 1, 1, -1)  # [1, 1, 1, num_kernels]

        diff_sq = (M_expanded - mu_expanded) ** 2
        kernel_values = torch.exp(-diff_sq / (2 * self.kernel_width ** 2))

        K = kernel_values.sum(dim=2)  # [batch_size, m, num_kernels]

        return K

    def prompt_attention(
        self,
        K: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply prompt attention to aggregate kernel pooling features.

        α_i = exp(u_i^T u_p) / Σ_j exp(u_j^T u_p)
        u_i = tanh(W_p · t_i^p + b_p)

        Args:
            K: [batch_size, m, num_kernels] - kernel pooling features
            prompt_embeds: [batch_size, m, embed_dim] - prompt embeddings
            prompt_mask: [batch_size, m] - prompt attention mask

        Returns:
            v_p: [batch_size, num_kernels * 3] - aggregated features (max, min, avg pooling)
        """
        u = torch.tanh(self.attention_transform(prompt_embeds))  # [batch_size, m, embed_dim]

        scores = torch.matmul(u, self.attention_context)  # [batch_size, m]

        scores = scores.masked_fill(prompt_mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=1)

        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [batch_size, m, 1]
        K_weighted = K * attention_weights_expanded

        features = []

        max_pool, _ = K.max(dim=1)  # [batch_size, num_kernels]
        features.append(max_pool)

        min_pool, _ = K.min(dim=1)  # [batch_size, num_kernels]
        features.append(min_pool)

        avg_pool = K_weighted.sum(dim=1)  # [batch_size, num_kernels]
        features.append(avg_pool)

        v_p = torch.cat(features, dim=-1)  # [batch_size, num_kernels * 3]

        return v_p

    def forward(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        essay_input_ids: torch.Tensor,
        essay_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            prompt_input_ids: [batch_size, m]
            prompt_attention_mask: [batch_size, m]
            essay_input_ids: [batch_size, n]
            essay_attention_mask: [batch_size, n]

        Returns:
            u_i: Prompt adherence features [batch_size, num_kernels * 3]
        """
        with torch.no_grad() if self.config.freeze_bert else torch.enable_grad():
            prompt_outputs = self.encoder(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                return_dict=True
            )
            essay_outputs = self.encoder(
                input_ids=essay_input_ids,
                attention_mask=essay_attention_mask,
                return_dict=True
            )

        prompt_embeds = prompt_outputs.last_hidden_state  # [batch_size, m, embed_dim]
        essay_embeds = essay_outputs.last_hidden_state    # [batch_size, n, embed_dim]

        M = self.compute_matching_matrix(
            prompt_embeds, essay_embeds,
            prompt_attention_mask, essay_attention_mask
        )

        K = self.kernel_pooling(M)

        u_i = self.prompt_attention(K, prompt_embeds, prompt_attention_mask)

        return u_i


class ESP(nn.Module):
    """
    Essay Score Predictor

    Combines EQ-net and PA-net features to predict essay score.
    ŷ_i = sigmoid(W_s × σ([v_i ⊕ u_i]) + b_s)
    """

    def __init__(self, config: PANNConfig, pa_output_dim: int):
        super().__init__()
        self.config = config

        input_dim = config.bert_hidden_size + pa_output_dim

        layers = []
        current_dim = input_dim

        for i in range(config.fc_layers):
            layers.append(nn.Linear(current_dim, config.fc_hidden_size))

            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            elif config.activation == "gelu":
                layers.append(nn.GELU())

            layers.append(nn.Dropout(config.dropout))
            current_dim = config.fc_hidden_size

        self.fc_layers = nn.Sequential(*layers)

        self.score_layer = nn.Linear(config.fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, v_i: torch.Tensor, u_i: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_i: [batch_size, bert_hidden_size] - quality features from EQ-net
            u_i: [batch_size, pa_output_dim] - adherence features from PA-net

        Returns:
            predictions: [batch_size] - predicted scores in [0, 1]
        """
        combined = torch.cat([v_i, u_i], dim=-1)  # [batch_size, input_dim]

        hidden = self.fc_layers(combined)

        logits = self.score_layer(hidden).squeeze(-1)
        predictions = self.sigmoid(logits)

        return predictions


class PromptAwareAES(nn.Module):
    """
    Complete PANN (Prompt-Aware Neural Network) model.

    Combines:
    - EQ-net: Essay quality features (prompt-invariant)
    - PA-net: Prompt adherence features (prompt-specific)
    - ESP: Score predictor

    Total parameters: ~112.52M (similar to paper)
    """

    def __init__(self, config: PANNConfig | None = None):
        super().__init__()
        self.config = config or PANNConfig()

        self.eq_net = EQNet(self.config)
        self.pa_net = PANet(self.config)
        self.esp = ESP(self.config, self.pa_net.output_dim)

        self.loss_fn = nn.MSELoss()

        print(f"PANN initialized with {self.count_parameters():,} parameters")

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PANN.

        Args:
            input_ids: [batch_size, seq_len] - essay tokens
            attention_mask: [batch_size, seq_len] - essay mask
            prompt_input_ids: [batch_size, prompt_len] - prompt tokens
            prompt_attention_mask: [batch_size, prompt_len] - prompt mask
            labels: [batch_size] - ground truth scores (normalized [0,1])

        Returns:
            Dict with 'predictions' and optionally 'loss'
        """
        v_i = self.eq_net(input_ids, attention_mask)

        u_i = self.pa_net(
            prompt_input_ids, prompt_attention_mask,
            input_ids, attention_mask
        )

        predictions = self.esp(v_i, u_i)

        outputs = {'predictions': predictions}

        if labels is not None:
            loss = self.loss_fn(predictions, labels)
            outputs['loss'] = loss

        outputs['eq_features'] = v_i
        outputs['pa_features'] = u_i

        return outputs

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference mode prediction.

        Args:
            input_ids: Essay tokens
            attention_mask: Essay mask
            prompt_input_ids: Prompt tokens
            prompt_attention_mask: Prompt mask

        Returns:
            predictions: [batch_size] - predicted scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids, attention_mask,
                prompt_input_ids, prompt_attention_mask
            )
            return outputs['predictions']


PANN = PromptAwareAES
