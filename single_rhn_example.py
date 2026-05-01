#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-layer Restricted Hopfield Network (RHN) example.

This script is reorganized from the notebook `A_Single_RHN_Example.ipynb`.
It demonstrates how to:
  1. load one MNIST sample from each digit class,
  2. binarize the samples into {-1, +1} patterns,
  3. train a single-layer RHN using the subspace rotation update,
  4. query the trained RHN until convergence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


torch.set_printoptions(precision=10)
torch.set_default_dtype(torch.float64)


@dataclass
class EarlyStopper:
    """Simple early stopping utility for RHN weight updates."""

    patience: int = 5
    min_delta: float = 0.0
    filename: str = "optimal_weight.pth"

    def __post_init__(self) -> None:
        self.counter = 0
        self.min_val_loss = np.inf

    def early_stop(self, model: "RHN", val_loss: torch.Tensor) -> bool:
        """
        Save the current weight when validation loss improves.
        Stop when loss increases for `patience` consecutive checks.
        """
        loss_value = float(val_loss.detach().cpu().item())

        if loss_value < self.min_val_loss:
            self.min_val_loss = loss_value
            torch.save(model.w_xh.detach().cpu(), self.filename)
            self.counter = 0
            return False

        if loss_value > self.min_val_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                loaded_weight = torch.load(self.filename, map_location=model.w_xh.device)
                model.w_xh = loaded_weight.to(dtype=model.w_xh.dtype, device=model.w_xh.device)
                return True

        return False


class RHN(nn.Module):
    """Single-layer Restricted Hopfield Network for illustration."""

    def __init__(
        self,
        inputnodes: int = 784,
        hiddennodes: int = 15,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.xnodes = inputnodes
        self.hnodes = hiddennodes
        self.dtype = dtype
        self.device = torch.device(device)

        # Orthogonal initialization. This keeps the original matrix shape.
        self.w_xh = torch.empty(self.xnodes, self.hnodes, dtype=dtype, device=self.device)
        nn.init.orthogonal_(self.w_xh)

        self.loss_lst: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.dtype, device=self.device)
        return torch.tanh(x @ self.w_xh) @ self.w_xh.T

    @staticmethod
    def loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_pred, y)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.dtype, device=self.device)
        h_output = x @ self.w_xh
        return -torch.linalg.norm(h_output @ h_output.T)

    @torch.no_grad()
    def subspace_train(
        self,
        inputs: torch.Tensor,
        early_stopper: Optional[EarlyStopper] = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Train RHN weights using a subspace-rotation-style update.

        The update follows the notebook logic:
            final_y = forward(inputs)
            trans_mat = inputs.T @ final_y
            U, S, Vh = svd(trans_mat)
            W <- U @ Vh @ W
        """
        inputs = inputs.to(dtype=self.dtype, device=self.device)

        for epoch in range(epochs):
            final_y = self.forward(inputs)
            loss_value = self.loss(inputs, final_y)
            self.loss_lst.append(float(loss_value.detach().cpu().item()))

            if verbose:
                print(f"Epoch {epoch:04d} | Loss: {loss_value.item():.10f}")

            if early_stopper is not None and early_stopper.early_stop(self, loss_value):
                if verbose:
                    print("Early stopping condition is satisfied.")
                return self.w_xh

            trans_mat = inputs.T @ final_y
            u, _, vh = torch.linalg.svd(trans_mat, full_matrices=False)

            # Keep the same training logic as the original notebook.
            self.w_xh = u @ vh @ self.w_xh

        return self.w_xh

    @torch.no_grad()
    def query(self, x: torch.Tensor, max_iter: int = 500, tol: float = 1e-2) -> torch.Tensor:
        """Iteratively query the RHN until the output becomes stable."""
        inputs = x.to(dtype=self.dtype, device=self.device)

        for _ in range(max_iter):
            final_y = torch.sign(self.forward(inputs))
            err = torch.linalg.norm(inputs - final_y)
            if err < tol:
                return final_y
            inputs = final_y

        return final_y


def load_mnist_digit_patterns(
    root: str = "./data",
    samples_per_digit: int = 1,
    binarize: bool = True,
) -> torch.Tensor:
    """
    Load MNIST and select `samples_per_digit` samples from each digit class.

    Returns
    -------
    patterns: Tensor of shape (10 * samples_per_digit, 784)
        Flattened MNIST patterns scaled to [-1, 1]. If `binarize=True`, values are
        converted to {-1, +1}.
    """
    train_data = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=None,
    )

    train_labels = train_data.targets
    train_images = train_data.data.to(torch.float64) / 255.0 * 2.0 - 1.0

    patterns = []
    for sample_idx in range(samples_per_digit):
        for digit in range(10):
            digit_indices = torch.where(train_labels == digit)[0]
            image = train_images[digit_indices[sample_idx]].reshape(784)
            patterns.append(image)

    patterns = torch.stack(patterns, dim=0)

    if binarize:
        patterns = torch.where(patterns < 0, -torch.ones_like(patterns), torch.ones_like(patterns))

    return patterns


def main() -> None:
    patterns = load_mnist_digit_patterns(root="./data", samples_per_digit=1, binarize=True)

    rhn = RHN(inputnodes=784, hiddennodes=30, dtype=torch.float64, device="cpu")
    early_stopper = EarlyStopper(patience=10, min_delta=0.0, filename="rhn_optimal_weight.pth")

    tic = time.perf_counter()
    weight_memory = rhn.subspace_train(patterns, early_stopper=early_stopper, epochs=10)
    toc = time.perf_counter()

    print(f"Training Time: {toc - tic:.4f} seconds")
    print(f"Learned weight shape: {tuple(weight_memory.shape)}")


if __name__ == "__main__":
    main()
