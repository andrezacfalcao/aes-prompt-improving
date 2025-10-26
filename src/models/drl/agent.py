"""
DRL-based AES Agent

Integrates CNAA and CST for disentangled representation learning.

Training pipeline:
1. CNAA Pre-training: Disentangle quality vs content in EQ-net
2. CST Fine-tuning: Disentangle quality vs adherence via counterfactuals

Implementação baseada no artigo ACL 2023:
"Improving Domain Generalization for Prompt-Aware Essay Scoring
via Disentangled Representation Learning"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .cnaa import CNAAConfig, CNAALoss, DataAugmenter
from .cst import CSTConfig, CSTLoss, CounterfactualGenerator, PreScoreGuidedTrainer


@dataclass
class DRLConfig:
    """Configuration for DRL training."""

    cnaa: CNAAConfig = None

    cst: CSTConfig = None

    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        if self.cnaa is None:
            self.cnaa = CNAAConfig()
        if self.cst is None:
            self.cst = CSTConfig()


class DRLAESAgent:
    """
    DRL-based AES Agent.

    Manages the complete training pipeline:
    1. CNAA pre-training of EQ-net
    2. PANN warmup training
    3. CST fine-tuning with counterfactual data
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: DRLConfig | None = None
    ):
        """
        Args:
            model: PANN model (PromptAwareAES)
            tokenizer: Tokenizer for text processing
            config: DRL configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DRLConfig()

        self.cnaa_loss = CNAALoss(self.config.cnaa)
        self.cst_loss = CSTLoss()
        self.data_augmenter = DataAugmenter(self.config.cnaa)
        self.cf_generator = CounterfactualGenerator(self.config.cst, tokenizer)
        self.psg_trainer = PreScoreGuidedTrainer(self.config.cst)

        self.model.to(self.config.device)
        self.cnaa_loss.to(self.config.device)

    def pretrain_cnaa(
        self,
        train_data: Dict[str, List],
        val_data: Optional[Dict[str, List]] = None
    ) -> Dict[str, List[float]]:
        """
        CNAA pre-training of EQ-net.

        Args:
            train_data: {
                'essays': List[str],
                'scores': List[float],
                'prompts': List[int]
            }
            val_data: Optional validation data

        Returns:
            training_history: Dict with loss curves
        """
        print("\n" + "="*50)
        print("CNAA Pre-training: Disentangling Quality vs Content")
        print("="*50)

        print("\nAugmenting dataset...")
        original_data, derived_data = self.data_augmenter.augment_dataset(
            train_data['essays'],
            train_data['scores'],
            train_data['prompts']
        )

        print(f"Original data: {len(original_data)} samples")
        print(f"Derived data: {len(derived_data)} samples")
        print(f"Total: {len(original_data) + len(derived_data)} samples")

        combined_data = original_data + derived_data

        optimizer = AdamW(
            self.model.eq_net.parameters(),
            lr=self.config.cnaa.pretrain_learning_rate
        )

        history = {'loss': [], 'nia_loss': [], 'asa_loss': []}
        self.model.eq_net.train()

        for epoch in range(self.config.cnaa.pretrain_epochs):
            total_loss = 0
            total_nia = 0
            total_asa = 0

            batch_size = 16
            n_batches = (len(combined_data) + batch_size - 1) // batch_size

            for i in range(n_batches):
                batch_data = combined_data[i*batch_size:(i+1)*batch_size]

                essays = [d['essay'] for d in batch_data]
                encodings = self.tokenizer(
                    essays,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )

                input_ids = encodings['input_ids'].to(self.config.device)
                attention_mask = encodings['attention_mask'].to(self.config.device)

                quality_labels = torch.tensor(
                    [d['quality'] for d in batch_data],
                    dtype=torch.long,
                    device=self.config.device
                )
                content_labels = torch.tensor(
                    [d['content'] for d in batch_data],
                    dtype=torch.long,
                    device=self.config.device
                )

                features = self.model.eq_net(input_ids, attention_mask)

                loss_dict = self.cnaa_loss(features, quality_labels, content_labels)
                loss = loss_dict['loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_nia += loss_dict['nia_loss']
                total_asa += loss_dict['asa_loss']

            avg_loss = total_loss / n_batches
            avg_nia = total_nia / n_batches
            avg_asa = total_asa / n_batches

            history['loss'].append(avg_loss)
            history['nia_loss'].append(avg_nia)
            history['asa_loss'].append(avg_asa)

            print(f"Epoch {epoch+1}/{self.config.cnaa.pretrain_epochs} - "
                  f"Loss: {avg_loss:.4f} (NIA: {avg_nia:.4f}, ASA: {avg_asa:.4f})")

        print("\nCNAA pre-training completed!")
        return history

    def warmup_training(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Warmup training: Train full PANN on original data.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader

        Returns:
            training_history: Dict with loss curves
        """
        print("\n" + "="*50)
        print("CST Warmup: Training PANN on Original Data")
        print("="*50)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.cst.warmup_learning_rate
        )

        history = {'loss': []}
        self.model.train()

        for epoch in range(self.config.cst.warmup_epochs):
            total_loss = 0
            n_batches = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                prompt_input_ids = batch['prompt_input_ids'].to(self.config.device)
                prompt_attention_mask = batch['prompt_attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    labels=labels
                )

                loss = outputs['loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            history['loss'].append(avg_loss)

            print(f"Epoch {epoch+1}/{self.config.cst.warmup_epochs} - Loss: {avg_loss:.4f}")

        print("\nWarmup training completed!")
        return history

    def generate_and_merge_counterfactuals(
        self,
        train_data: Dict[str, List]
    ) -> List[Dict]:
        """
        Generate counterfactual data and merge scores.

        Args:
            train_data: Original training data

        Returns:
            updated_cf_data: Counterfactual data with merged scores
        """
        print("\n" + "="*50)
        print("Generating Counterfactual Data")
        print("="*50)

        cf_data = self.cf_generator.create_counterfactual_dataset(
            train_data['prompts_text'],
            train_data['essays'],
            train_data['scores']
        )

        print(f"Generated {len(cf_data)} counterfactual samples")

        print("\nGenerating pseudo-scores...")
        pseudo_scores = self.psg_trainer.generate_pseudo_scores(
            self.model,
            cf_data,
            self.tokenizer,
            self.config.device
        )

        print("Merging pre-scores and pseudo-scores...")
        updated_cf_data = self.psg_trainer.update_counterfactual_scores(
            cf_data,
            pseudo_scores
        )

        print("Counterfactual data ready!")
        return updated_cf_data

    def finetune_cst(
        self,
        train_loader: DataLoader,
        cf_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        CST fine-tuning with counterfactual data.

        Args:
            train_loader: Original training data
            cf_loader: Counterfactual data loader
            val_loader: Optional validation loader

        Returns:
            training_history: Dict with loss curves
        """
        print("\n" + "="*50)
        print("CST Fine-tuning: Training with Counterfactual Data")
        print("="*50)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.cst.finetune_learning_rate
        )

        history = {'loss': [], 'original_loss': [], 'cf_loss': []}
        self.model.train()

        for epoch in range(self.config.cst.finetune_epochs):
            total_loss = 0
            total_original_loss = 0
            total_cf_loss = 0
            n_batches = 0

            for original_batch, cf_batch in zip(train_loader, cf_loader):
                input_ids = original_batch['input_ids'].to(self.config.device)
                attention_mask = original_batch['attention_mask'].to(self.config.device)
                prompt_input_ids = original_batch['prompt_input_ids'].to(self.config.device)
                prompt_attention_mask = original_batch['prompt_attention_mask'].to(self.config.device)
                labels = original_batch['labels'].to(self.config.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    labels=labels
                )

                original_loss = outputs['loss']

                cf_input_ids = cf_batch['input_ids'].to(self.config.device)
                cf_attention_mask = cf_batch['attention_mask'].to(self.config.device)
                cf_prompt_input_ids = cf_batch['prompt_input_ids'].to(self.config.device)
                cf_prompt_attention_mask = cf_batch['prompt_attention_mask'].to(self.config.device)
                cf_labels = cf_batch['merged_scores'].to(self.config.device)  # Use merged scores!

                cf_outputs = self.model(
                    input_ids=cf_input_ids,
                    attention_mask=cf_attention_mask,
                    prompt_input_ids=cf_prompt_input_ids,
                    prompt_attention_mask=cf_prompt_attention_mask,
                    labels=cf_labels
                )

                cf_loss = cf_outputs['loss']

                loss = original_loss + cf_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_original_loss += original_loss.item()
                total_cf_loss += cf_loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            avg_original = total_original_loss / n_batches
            avg_cf = total_cf_loss / n_batches

            history['loss'].append(avg_loss)
            history['original_loss'].append(avg_original)
            history['cf_loss'].append(avg_cf)

            print(f"Epoch {epoch+1}/{self.config.cst.finetune_epochs} - "
                  f"Loss: {avg_loss:.4f} (Original: {avg_original:.4f}, CF: {avg_cf:.4f})")

        print("\nCST fine-tuning completed!")
        return history

    def train(
        self,
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
        enable_cnaa: bool = True,
        enable_cst: bool = True
    ) -> Dict[str, Any]:
        """
        Complete DRL training pipeline.

        Args:
            train_data: Training data
            val_data: Optional validation data
            enable_cnaa: Whether to run CNAA pre-training
            enable_cst: Whether to run CST fine-tuning

        Returns:
            training_history: Complete training history
        """
        history = {}

        if enable_cnaa:
            cnaa_history = self.pretrain_cnaa(train_data, val_data)
            history['cnaa'] = cnaa_history
        else:
            print("\nSkipping CNAA pre-training...")

        if enable_cst:
            warmup_history = self.warmup_training(
                train_data['train_loader'],
                val_data.get('val_loader') if val_data else None
            )
            history['warmup'] = warmup_history

            cf_data = self.generate_and_merge_counterfactuals(train_data)
        else:
            print("\nSkipping CST fine-tuning...")

        return history

    def evaluate(
        self,
        test_loader: DataLoader,
        asap_dataset = None,
        prompt_id: int = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test set with full metrics.

        Args:
            test_loader: Test data loader
            asap_dataset: ASAPDataset instance for denormalization (optional)
            prompt_id: Prompt ID for score denormalization (optional)

        Returns:
            metrics: Dict with evaluation metrics (QWK, Pearson, Spearman, RMSE, MAE)
        """
        from src.evaluation.metrics import evaluate_predictions

        self.model.eval()

        all_predictions = []
        all_labels = []
        all_prompt_ids = []

        print("\nEvaluating model...")

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                prompt_input_ids = batch['prompt_input_ids'].to(self.config.device)
                prompt_attention_mask = batch['prompt_attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask
                )

                predictions = outputs['predictions']

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if 'prompt_id' in batch:
                    all_prompt_ids.extend(batch['prompt_id'].cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        if asap_dataset is not None and prompt_id is not None:
            print(f"Denormalizing scores for prompt {prompt_id}...")
            all_predictions_denorm = np.array([
                asap_dataset.denormalize_score(pred, prompt_id)
                for pred in all_predictions
            ])
            all_labels_denorm = np.array([
                asap_dataset.denormalize_score(label, prompt_id)
                for label in all_labels
            ])

            metrics = evaluate_predictions(
                all_labels_denorm,
                all_predictions_denorm,
                round_for_qwk=True
            )

            norm_metrics = evaluate_predictions(
                all_labels,
                all_predictions,
                round_for_qwk=False,
                prefix="norm_"
            )
            metrics.update(norm_metrics)

        else:
            metrics = evaluate_predictions(
                all_labels,
                all_predictions,
                round_for_qwk=False
            )

        print("Evaluation completed!")

        return metrics

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
