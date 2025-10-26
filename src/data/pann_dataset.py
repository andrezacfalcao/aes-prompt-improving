"""
PANN Dataset loader - includes prompt text tokenization

PANN requires both essay and prompt tokens as input:
- Essay tokens (from essay text)
- Prompt tokens (from prompt question/description)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict


ASAP_PROMPTS = {
    1: """Write about an experience that changed your perspective or taught you something about yourself or the world.
    Describe the experience and explain how it affected you.""",

    2: """Write about whether or not you believe censorship of books and other materials is justified.
    Use examples from your reading or experience to support your position.""",

    3: """Write a story about a time when you had to be patient.
    Describe what happened and how you felt.""",

    4: """Write about a time when you learned something new.
    Describe what you learned and how the experience affected you.""",

    5: """Describe a person who has influenced you.
    Explain how and why this person has been important to you.""",

    6: """Write about a time when you helped someone or someone helped you.
    Describe what happened and explain how this experience affected you.""",

    7: """Write about an event that you think is important.
    Explain why the event is significant and how it has influenced you or others.""",

    8: """Write about the effects of computers on modern life.
    Discuss both positive and negative effects, and explain your opinion."""
}


class PANNDataset(Dataset):
    """
    PyTorch Dataset for PANN model.

    Returns both essay and prompt tokens for PA-net to compute
    prompt adherence features.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_essay_length: int = 512,
        max_prompt_length: int = 128,
        prompt_texts: Optional[Dict[int, str]] = None
    ):
        """
        Args:
            dataframe: DataFrame with columns 'essay', 'normalized_score', 'essay_set'
            tokenizer: Tokenizer (from transformers)
            max_essay_length: Maximum sequence length for essays
            max_prompt_length: Maximum sequence length for prompts
            prompt_texts: Optional dict mapping prompt_id -> prompt text
                         (defaults to ASAP_PROMPTS)
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_essay_length = max_essay_length
        self.max_prompt_length = max_prompt_length
        self.prompt_texts = prompt_texts or ASAP_PROMPTS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        essay = row['essay']
        score = row['normalized_score']
        prompt_id = row['essay_set']

        prompt_text = self.prompt_texts.get(prompt_id, "")

        essay_encoding = self.tokenizer(
            essay,
            max_length=self.max_essay_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        prompt_encoding = self.tokenizer(
            prompt_text,
            max_length=self.max_prompt_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': essay_encoding['input_ids'].squeeze(0),
            'attention_mask': essay_encoding['attention_mask'].squeeze(0),

            'prompt_input_ids': prompt_encoding['input_ids'].squeeze(0),
            'prompt_attention_mask': prompt_encoding['attention_mask'].squeeze(0),

            'labels': torch.tensor(score, dtype=torch.float32),
            'score': torch.tensor(score, dtype=torch.float32),
            'prompt_id': torch.tensor(prompt_id, dtype=torch.long),

            'essay_id': row['essay_id'],
            'essay_text': essay,
            'prompt_text': prompt_text
        }


class PANNDataModule:
    """
    Data module for PANN training.

    Handles loading, preprocessing, and creating data loaders
    with proper prompt tokenization.
    """

    def __init__(
        self,
        asap_dataset,
        tokenizer,
        max_essay_length: int = 512,
        max_prompt_length: int = 128,
        batch_size: int = 8,
        num_workers: int = 0
    ):
        """
        Args:
            asap_dataset: ASAPDataset instance (already loaded)
            tokenizer: Tokenizer for text encoding
            max_essay_length: Max essay sequence length
            max_prompt_length: Max prompt sequence length
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
        """
        self.asap_dataset = asap_dataset
        self.tokenizer = tokenizer
        self.max_essay_length = max_essay_length
        self.max_prompt_length = max_prompt_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def create_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Create PyTorch data loaders for train/val/test.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe

        Returns:
            Dict with 'train', 'val', 'test' data loaders
        """
        from torch.utils.data import DataLoader

        train_dataset = PANNDataset(
            train_df,
            self.tokenizer,
            self.max_essay_length,
            self.max_prompt_length
        )

        val_dataset = PANNDataset(
            val_df,
            self.tokenizer,
            self.max_essay_length,
            self.max_prompt_length
        )

        test_dataset = PANNDataset(
            test_df,
            self.tokenizer,
            self.max_essay_length,
            self.max_prompt_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        print(f"\nData loaders created:")
        print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
        print(f"  Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
        print(f"  Test: {len(test_dataset)} samples ({len(test_loader)} batches)")

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    def prepare_for_drl(
        self,
        train_df: pd.DataFrame
    ) -> Dict:
        """
        Prepare data for DRL training (CNAA/CST).

        Returns data in format needed by DRL agent:
        - Raw text lists for augmentation
        - Normalized scores
        - Prompt IDs and texts

        Args:
            train_df: Training dataframe

        Returns:
            Dict with lists of essays, scores, prompts, etc.
        """
        return {
            'essays': train_df['essay'].tolist(),
            'scores': train_df['normalized_score'].tolist(),
            'prompts': train_df['essay_set'].tolist(),
            'prompts_text': [ASAP_PROMPTS.get(p, "") for p in train_df['essay_set'].tolist()],
            'essay_ids': train_df['essay_id'].tolist()
        }
