"""
ASAP Dataset loader and preprocessing
"""

import pandas as pd
import re
from typing import List, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class ASAPDataset:
    """
    ASAP (Automated Student Assessment Prize) Dataset handler

    O dataset ASAP contém 8 prompts diferentes com essays e scores.
    """

    SCORE_RANGES = {
        1: (2, 12),
        2: (1, 6),
        3: (0, 3),
        4: (0, 3),
        5: (0, 4),
        6: (0, 4),
        7: (2, 24),
        8: (10, 60),
    }

    def __init__(self, data_path: str, use_columns: Optional[List[str]] = None):
        """
        Args:
            data_path: Path to ASAP TSV file
            use_columns: Colunas a utilizar (default: essay_id, essay_set, essay, domain1_score)
        """
        self.data_path = Path(data_path)
        self.use_columns = use_columns or ['essay_id', 'essay_set', 'essay', 'domain1_score']
        self.data = None
        self.score_stats = {}

    def load_data(self) -> pd.DataFrame:
        """Load ASAP dataset from disk"""
        print(f"Loading ASAP dataset from {self.data_path}")

        self.data = pd.read_csv(
            self.data_path,
            sep='\t',
            encoding='latin-1',
            usecols=self.use_columns
        )

        self.data = self.data.dropna(subset=['domain1_score'])

        for prompt_id in self.data['essay_set'].unique():
            prompt_data = self.data[self.data['essay_set'] == prompt_id]
            self.score_stats[prompt_id] = {
                'min': prompt_data['domain1_score'].min(),
                'max': prompt_data['domain1_score'].max(),
                'mean': prompt_data['domain1_score'].mean(),
                'std': prompt_data['domain1_score'].std(),
                'count': len(prompt_data)
            }

        print(f"Loaded {len(self.data)} essays across {len(self.score_stats)} prompts")
        return self.data

    def preprocess_text(self, text: str) -> str:
        """
        Pré-processamento básico de texto

        Args:
            text: Texto bruto do essay

        Returns:
            Texto pré-processado
        """
        if not isinstance(text, str):
            return ""

        text = text.strip('"')

        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'\s+([.,!?])', r'\1', text)

        return text.strip()

    def normalize_score(self, score: float, prompt_id: int) -> float:
        """
        Normaliza score para range [0,1]

        Args:
            score: Score original
            prompt_id: ID do prompt (1-8)

        Returns:
            Score normalizado [0,1]
        """
        min_score, max_score = self.SCORE_RANGES[prompt_id]
        return (score - min_score) / (max_score - min_score)

    def denormalize_score(self, norm_score: float, prompt_id: int) -> float:
        """
        Desnormaliza score de [0,1] para range original

        Args:
            norm_score: Score normalizado [0,1]
            prompt_id: ID do prompt (1-8)

        Returns:
            Score no range original
        """
        min_score, max_score = self.SCORE_RANGES[prompt_id]
        return norm_score * (max_score - min_score) + min_score

    def get_cross_prompt_split(
        self,
        source_prompts: List[int],
        target_prompt: int,
        val_size: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Cria split cross-prompt para treinamento e teste

        Args:
            source_prompts: Lista de prompts para treino (ex: [1, 3, 4, 5, 6, 7, 8])
            target_prompt: Prompt para teste (ex: 2)
            val_size: Proporção dos dados de treino para validação
            random_seed: Seed para reprodutibilidade

        Returns:
            (train_df, val_df, test_df)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        train_val_data = self.data[self.data['essay_set'].isin(source_prompts)].copy()

        test_data = self.data[self.data['essay_set'] == target_prompt].copy()

        if val_size > 0:
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_size,
                random_state=random_seed,
                stratify=train_val_data['essay_set']  
            )
        else:
            train_data = train_val_data
            val_data = pd.DataFrame(columns=train_val_data.columns)

        for df in [train_data, val_data, test_data]:
            if len(df) > 0:
                df['essay'] = df['essay'].apply(self.preprocess_text)
                df['normalized_score'] = df.apply(
                    lambda row: self.normalize_score(row['domain1_score'], row['essay_set']),
                    axis=1
                )

        print(f"Cross-prompt split:")
        print(f"  Train prompts: {source_prompts} ({len(train_data)} essays)")
        print(f"  Val prompts: {source_prompts} ({len(val_data)} essays)")
        print(f"  Test prompt: {target_prompt} ({len(test_data)} essays)")

        return train_data, val_data, test_data


class EssayDataset(Dataset):
    """
    PyTorch Dataset para essays
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            dataframe: DataFrame com colunas 'essay', 'normalized_score', 'essay_set'
            tokenizer: Tokenizer (de transformers)
            max_length: Comprimento máximo da sequência
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        essay = row['essay']
        score = row['normalized_score']
        prompt_id = row['essay_set']

        encoding = self.tokenizer(
            essay,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'score': torch.tensor(score, dtype=torch.float32),
            'prompt_id': torch.tensor(prompt_id, dtype=torch.long),
            'essay_id': row['essay_id']
        }
