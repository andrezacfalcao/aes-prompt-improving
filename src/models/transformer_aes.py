"""
Transformer-based models for Automated Essay Scoring
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional, Dict


class TransformerAES(nn.Module):
    """
    Transformer-based Automated Essay Scoring model (Baseline)

    Arquitetura:
    - Transformer encoder (BERT, RoBERTa, etc.)
    - Pooling do [CLS] token
    - Dropout
    - Linear layer para regressão (output: score normalizado [0,1])

    Suporta diferentes backbones:
    - BERT
    - RoBERTa
    - DistilBERT
    - ELECTRA
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 1,
        dropout: float = 0.1,
        hidden_size: Optional[int] = None,
        freeze_encoder: bool = False
    ):
        """
        Args:
            model_name: Pretrained model name (e.g., 'bert-base-uncased')
            num_labels: Number of output labels (1 for regression)
            dropout: Dropout rate
            hidden_size: Hidden layer size (None = use encoder hidden size)
            freeze_encoder: Se deve congelar o encoder durante treinamento
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        print(f"Loading transformer model: {model_name}")

        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        encoder_hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        if hidden_size is not None:
            self.hidden_layer = nn.Linear(encoder_hidden_size, hidden_size)
            self.activation = nn.ReLU()
            self.score_layer = nn.Linear(hidden_size, num_labels)
        else:
            self.hidden_layer = None
            self.score_layer = nn.Linear(encoder_hidden_size, num_labels)

        self.sigmoid = nn.Sigmoid()

        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth scores [batch_size] (normalized [0,1])

        Returns:
            Dict com 'loss' (se labels fornecido) e 'predictions'
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        pooled_output = self.dropout(pooled_output)

        if self.hidden_layer is not None:
            hidden = self.hidden_layer(pooled_output)
            hidden = self.activation(hidden)
            hidden = self.dropout(hidden)
        else:
            hidden = pooled_output

        logits = self.score_layer(hidden)  # [batch_size, 1]
        logits = logits.squeeze(-1)  # [batch_size]

        predictions = self.sigmoid(logits)

        outputs = {'predictions': predictions}

        if labels is not None:
            loss = self.loss_fn(predictions, labels)
            outputs['loss'] = loss

        return outputs

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predição simples (sem loss)

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask

        Returns:
            Predictions [batch_size]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs['predictions']


class BertAES(TransformerAES):
    """
    BERT-based AES model

    Usa bert-base-uncased como backbone
    """

    def __init__(
        self,
        dropout: float = 0.1,
        hidden_size: Optional[int] = None,
        freeze_encoder: bool = False
    ):
        super().__init__(
            model_name="bert-base-uncased",
            num_labels=1,
            dropout=dropout,
            hidden_size=hidden_size,
            freeze_encoder=freeze_encoder
        )


class RobertaAES(TransformerAES):
    """
    RoBERTa-based AES model

    Usa roberta-base como backbone
    """

    def __init__(
        self,
        dropout: float = 0.1,
        hidden_size: Optional[int] = None,
        freeze_encoder: bool = False
    ):
        super().__init__(
            model_name="roberta-base",
            num_labels=1,
            dropout=dropout,
            hidden_size=hidden_size,
            freeze_encoder=freeze_encoder
        )


class DistilBertAES(TransformerAES):
    """
    DistilBERT-based AES model

    Versão mais leve e rápida (66M parâmetros vs 110M do BERT)
    """

    def __init__(
        self,
        dropout: float = 0.1,
        hidden_size: Optional[int] = None,
        freeze_encoder: bool = False
    ):
        super().__init__(
            model_name="distilbert-base-uncased",
            num_labels=1,
            dropout=dropout,
            hidden_size=hidden_size,
            freeze_encoder=freeze_encoder
        )


class ElectraAES(TransformerAES):
    """
    ELECTRA-based AES model

    Usa google/electra-base-discriminator
    """

    def __init__(
        self,
        dropout: float = 0.1,
        hidden_size: Optional[int] = None,
        freeze_encoder: bool = False
    ):
        super().__init__(
            model_name="google/electra-base-discriminator",
            num_labels=1,
            dropout=dropout,
            hidden_size=hidden_size,
            freeze_encoder=freeze_encoder
        )
