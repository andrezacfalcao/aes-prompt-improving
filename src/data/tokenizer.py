"""
Tokenization utilities for different transformer models
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional


class EssayDataset(Dataset):
    """Dataset customizado que retorna dicionários"""

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


def get_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    """
    Carrega tokenizer para um modelo específico

    Args:
        model_name: Nome do modelo (ex: 'bert-base-uncased')
        cache_dir: Diretório para cache (opcional)

    Returns:
        Tokenizer do transformers
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    return tokenizer


class EssayTokenizer:
    """
    Wrapper simplificado para tokenização de essays
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        """
        Args:
            model_name: Nome do modelo pré-treinado ou objeto tokenizer já carregado
            max_length: Comprimento máximo da sequência
        """
        self.max_length = max_length

        if isinstance(model_name, str):
            self.model_name = model_name
            self.tokenizer = self.load_tokenizer()
        else:
            self.tokenizer = model_name
            self.model_name = getattr(model_name, 'name_or_path', 'unknown')

    def load_tokenizer(self):
        """Carrega tokenizer"""
        return get_tokenizer(self.model_name)

    def __call__(self, *args, **kwargs):
        """
        Passa chamadas diretamente para o tokenizer
        """
        return self.tokenizer(*args, **kwargs)

    def tokenize_batch(
        self,
        texts: List[str],
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict:
        """
        Tokeniza um batch de essays

        Args:
            texts: Lista de textos
            padding: Estratégia de padding
            truncation: Se deve truncar
            return_tensors: Formato de retorno ('pt' para PyTorch)

        Returns:
            Dict com input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )

    def tokenize_dataset(
        self,
        texts: List[str],
        labels: List[float],
        padding: str = 'max_length',
        truncation: bool = True
    ) -> Dataset:
        """
        Tokeniza um dataset completo e retorna um Dataset do PyTorch

        Args:
            texts: Lista de textos (essays)
            labels: Lista de labels (scores)
            padding: Estratégia de padding
            truncation: Se deve truncar

        Returns:
            EssayDataset com input_ids, attention_mask e labels
        """
        # Tokeniza todos os textos
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='pt'
        )

        # Converte labels para tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # Cria EssayDataset
        dataset = EssayDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            labels_tensor
        )

        return dataset
