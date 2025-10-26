"""
Tokenization utilities for different transformer models
"""

from transformers import AutoTokenizer
from typing import List, Dict, Optional


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
            model_name: Nome do modelo pré-treinado
            max_length: Comprimento máximo da sequência
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = self.load_tokenizer()

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
