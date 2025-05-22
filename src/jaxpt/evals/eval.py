from typing import Callable

import numpy as np
import flax.nnx as nnx

from jaxpt.dataloaders import DataLoader


def calc_validation_loss(
    model: nnx.Module, loss_fn: Callable, dataloader: DataLoader, eval_steps=10
):
    valid_loss = 0.0
    for _ in range(eval_steps):
        batch, targets = dataloader()
        batch = np.squeeze(batch)
        targets = np.squeeze(targets)
        loss = loss_fn(model, batch, targets)
        valid_loss += loss
    valid_loss /= eval_steps
    return valid_loss
