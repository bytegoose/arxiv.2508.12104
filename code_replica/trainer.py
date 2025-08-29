"""trainer.py

This module defines the Trainer class that manages the training loop for the
COMET model. It performs the forward and backward passes using the AdamW optimizer
with a cosine learning rate decay schedule (with a target decay factor of 10x),
computes the cross-entropy loss for next-token prediction, and handles checkpointing.

The Trainer accepts:
  - A model instance (from model.py)
  - A training dataset (from DatasetLoader; assumed to be a PyTorch Dataset yielding tokenized sequences)
  - A configuration dictionary (derived from config.yaml)

It uses torch, torch.nn, torch.optim, and torch.utils.data.DataLoader for batching,
and utilizes tqdm for progress tracking. All hyperparameters default to safe values
if not provided in the configuration.
"""

import os
import math
import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup module-level logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Trainer:
    """
    Trainer is responsible for training the COMET model.

    Attributes:
        model (nn.Module): The decoder-only transformer model.
        train_dataset (torch.utils.data.Dataset): Training dataset yielding token sequences.
        config (Dict[str, Any]): Configuration dictionary containing hyperparameters.
        device (torch.device): Device on which training is performed.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs for training.
        optimizer (torch.optim.Optimizer): Optimizer, AdamW in this case.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Cosine annealing learning rate scheduler.
        criterion (nn.Module): CrossEntropy loss function.
        train_loader (DataLoader): DataLoader over the training dataset.
        global_step (int): Overall training step count.
        current_epoch (int): Current epoch number.
    """

    def __init__(self, model: nn.Module, train_dataset: Any, config: Dict[str, Any]) -> None:
        """
        Initialize the Trainer with the model, training dataset, and configuration.

        Args:
            model (nn.Module): The COMET model.
            train_dataset (Any): Training dataset object (should be compatible with PyTorch DataLoader).
            config (Dict[str, Any]): Configuration dictionary.
        """
        # Set device (GPU if available)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = model.to(self.device)
        self.config: Dict[str, Any] = config

        # Set training hyperparameters from config with defaults.
        training_config: Dict[str, Any] = self.config.get("training", {})
        self.batch_size: int = int(training_config.get("batch_size", 512))
        # Default number of epochs: if not specified or set to a non-numeric value, use 10 epochs.
        try:
            self.num_epochs: int = int(training_config.get("num_epochs", 10))
        except (ValueError, TypeError):
            self.num_epochs = 10

        self.learning_rate: float = float(training_config.get("learning_rate", 1e-4))
        # Use a default weight decay value if not specified.
        self.weight_decay: float = float(training_config.get("weight_decay", 0.01))

        # Create a DataLoader from the training dataset.
        self.train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.default_collate_fn  # Using our own default collate function.
        )

        # Initialize the optimizer (AdamW)
        self.optimizer: optim.Optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Determine the total number of training steps to set for the scheduler.
        total_steps: int = self.num_epochs * len(self.train_loader)
        # Scheduler: Cosine Annealing LR such that lr decays by 10x over the full training run.
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=self.learning_rate / 10
        )

        # Loss function: CrossEntropyLoss for next-token prediction.
        self.criterion: nn.Module = nn.CrossEntropyLoss()

        # Training state variables.
        self.global_step: int = 0
        self.current_epoch: int = 0

        logging.info("Trainer initialized on device: %s", self.device)
        logging.info("Batch size: %d, Num epochs: %d, Total training steps: %d", self.batch_size, self.num_epochs, total_steps)

    @staticmethod
    def default_collate_fn(batch):
        """
        Default collate function to process a batch of examples.
        Assumes each element in the batch is a list of token IDs and pads sequences to the maximum length in the batch.
        
        Args:
            batch (List[List[int]]): A list of token ID lists.
        
        Returns:
            torch.Tensor: A tensor of shape [batch_size, max_seq_length] with padded token IDs.
        """
        # Determine max sequence length in the batch.
        max_length: int = max(len(item) for item in batch)
        # Pad sequences with 0 (assumed to be the UNK or PAD token)
        padded_batch = []
        for item in batch:
            padded_item = item + [0] * (max_length - len(item))
            padded_batch.append(padded_item)
        return torch.tensor(padded_batch, dtype=torch.long)

    def train(self) -> None:
        """
        Run the training loop over the training dataset.
        For each epoch and each mini-batch, performs:
            - Forward pass on input tokens (excluding the last token).
            - Compute next-token prediction loss using shifted targets.
            - Backward propagation, gradient clipping, optimizer step, and scheduler step.
            - Logging of training loss and learning rate.
        Optionally saves checkpoints at the end of each epoch.
        """
        self.model.train()
        logging.info("Starting training loop for %d epochs.", self.num_epochs)

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_loss: float = 0.0
            batch_count: int = 0

            # Use tqdm for progress bar over batches.
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for batch in progress_bar:
                # Ensure gradients are enabled.
                torch.set_grad_enabled(True)

                # The batch is assumed to be a tensor of token IDs [B, seq_length]
                inputs: torch.Tensor = batch.to(self.device)

                # Ensure that sequence length is at least 2 (for input and target shifting)
                if inputs.size(1) < 2:
                    continue

                # Split into inputs and targets; target is inputs shifted by one position.
                input_seq: torch.Tensor = inputs[:, :-1]       # [B, T-1]
                target_seq: torch.Tensor = inputs[:, 1:]         # [B, T-1]

                # Forward pass: obtain logits; logits shape: [B, T-1, vocab_size]
                logits: torch.Tensor = self.model(input_seq)

                # Reshape logits and targets for cross-entropy loss.
                B, T_minus_one, vocab_size = logits.shape
                logits_flat: torch.Tensor = logits.reshape(B * T_minus_one, vocab_size)
                targets_flat: torch.Tensor = target_seq.reshape(B * T_minus_one)

                loss: torch.Tensor = self.criterion(logits_flat, targets_flat)
                loss_value: float = loss.item()

                # Backward pass.
                loss.backward()

                # Optional: Gradient clipping to avoid exploding gradients.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step.
                self.optimizer.step()
                # Zero out gradients.
                self.optimizer.zero_grad()

                # Scheduler step.
                self.scheduler.step()
                self.global_step += 1

                epoch_loss += loss_value
                batch_count += 1

                # Log progress with current training loss and learning rate.
                current_lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(loss=loss_value, lr=current_lr)

            avg_epoch_loss: float = epoch_loss / max(batch_count, 1)
            logging.info("Epoch %d completed: Average Loss = %.4f, Global Step = %d", epoch + 1, avg_epoch_loss, self.global_step)
            logging.info("Learning rate at end of epoch: %.6f", self.optimizer.param_groups[0]["lr"])

            # Optionally save checkpoint at the end of each epoch.
            checkpoint_path: str = f"checkpoint_epoch_{epoch + 1}.pt"
            self.save_checkpoint(checkpoint_path)
            logging.info("Checkpoint saved to %s", checkpoint_path)

    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint of the current training state.

        The checkpoint contains:
            - current_epoch
            - global_step
            - model state_dict
            - optimizer state_dict
            - scheduler state_dict

        Args:
            path (str): File path to save the checkpoint.
        """
        checkpoint: Dict[str, Any] = {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }
        torch.save(checkpoint, path)
        logging.info("Checkpoint successfully saved at %s", path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint from the given file path.

        Loads state into model, optimizer, scheduler, and updates training state variables.

        Args:
            path (str): File path containing the checkpoint.
        """
        if not os.path.exists(path):
            logging.error("Checkpoint file %s does not exist.", path)
            raise FileNotFoundError(f"Checkpoint file {path} not found.")
        
        checkpoint: Dict[str, Any] = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get("model_state_dict", {}))
        self.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
        self.scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", {}))
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        logging.info("Checkpoint loaded from %s; Resuming from epoch %d, global step %d", path, self.current_epoch, self.global_step)
