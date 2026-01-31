#!/usr/bin/env python
# coding: utf-8
# Let's Train Tiny MoE
# Author: Vikram Pawar (pvikram035 [at] gmail [dot] com)

import os
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat
from dataclasses import dataclass

from matplotlib import pyplot as plt

import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import optax

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Add jaxpt to path
import sys
jaxpt_dir = str(Path(__file__).absolute().parent.parent / "src")
sys.path.insert(0, jaxpt_dir)

from jaxpt.models import Tiny_MoE_2_Config, Tiny_MoE_2
from jaxpt.dataloaders import HuggingfaceDataLoader
from jaxpt.utils import (
    generate_readable_code,
    count_params,
    create_sharded_model,
    append_to_csv,
)
from jaxpt.checkpointers import save_checkpoint, save_optimizer_state

logger.info(f"Jax Version: {jax.__version__}")
logger.info(f"Flax Version: {flax.__version__}")

# Set up logging
output_dir = Path("training_runs").absolute()
timestamp = datetime.now().strftime("%Y%m%d")
random_code = generate_readable_code()
run_name = f"run_{timestamp}_{random_code}"
log_dir = output_dir / "Tiny_MoE_2" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Run: {run_name}")
logger.info(f"Log directory: {log_dir}")
logger.info(f"Output dir: {output_dir}")

jax.print_environment_info()
logger.info(f"Optax version: {optax.__version__}")
device = jax.default_backend()
logger.info(f"Platform: {device}")
devices = jax.devices()
num_devices = len(jax.devices())
logger.info(f"Num Devices: {num_devices}")
logger.info(f"Devices: {jax.devices()}")


# Set the default precision for matrix multiplication
#####################################
##        jax.lax matmul presets   ##
#####################################
## 'ANY_F8_ANY_F8_F32',
## 'ANY_F8_ANY_F8_F32_FAST_ACCUM'
## 'ANY_F8_ANY_F8_ANY'
## 'ANY_F8_ANY_F8_ANY_FAST_ACCUM'
## 'F16_F16_F16'
## 'F16_F16_F32'
## 'BF16_BF16_BF16'
## 'BF16_BF16_F32'
## 'BF16_BF16_F32_X3'
## 'BF16_BF16_F32_X6'
## 'TF32_TF32_F32'
## 'TF32_TF32_F32_X3'
## 'F32_F32_F32'
## 'F64_F64_F64'
#####################################
jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32")

# Create model

rngs = nnx.Rngs(default=jax.random.key(1337), gate_noise=jax.random.key(42))

config = Tiny_MoE_2_Config(
    name="Tiny_MoE_2",
    dtype=jnp.bfloat16,
    vocab_size=50304,
    n_layer=30,
    block_size=2048,
    n_head=12,
    n_kv_head=4,
    n_embed=672,
    n_mlp_hidden=2048,
    moe_bias=True,
    mlp_bias=False,
    attention_bias=False,
    load_balance_loss_coeff=1e-2,
    z_loss_coeff=5e-4,
    load_factor=1.25,
    ln_epsilon=1e-5,
    use_qk_norm=True,
    sdpa_implementation="cudnn" if device == "gpu" else "xla",
)
logger.info(f"Model config:\n{pformat(config)}")

mesh = Mesh(jax.devices(), ["devices"])
with mesh:
    m = create_sharded_model(Tiny_MoE_2, config, rngs)
    graphdef, rngstate, state = nnx.split(m, nnx.RngState, ...)
    total_params = count_params(m)
    moe_params = count_params(m, "moe")

logger.info(f"Parameter Count: {total_params:,}")
logger.info(f"Sharded / MoE Parameter Count: {moe_params:,}")
logger.info(f"Replicated Parameter Count: {total_params - moe_params:,}")

# Set up training config


@dataclass
class TrainerConfig:
    num_tokens: int = int(100e9)
    num_tokens_per_batch: int = 2**18  # 2**20 = 1.0 million
    mB: int = 16 * num_devices
    T: int = config.block_size
    max_steps: int = int(num_tokens // num_tokens_per_batch)
    max_lr: float = 1e-3
    min_lr: float = max_lr * 0.1
    max_grad_norm: float = 1.0  # Clip gradients to this norm
    weight_decay: float = 0.1  # Weight decay for adamw
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    warmup_steps: int = max_steps // 100
    print_interval: int = 100
    val: bool = True
    val_interval: int = 5000
    val_batches: int = 50  # Number of batches to use for validation
    checkpoint_model: bool = True
    checkpoint_optimizer: bool = False
    checkpoint_interval: int = 10000


trconf = TrainerConfig()


# Set up optimizer
def inverse_sqrt_schedule(step):
    warmup_lr = trconf.max_lr * (step + 1) / trconf.warmup_steps
    regular_lr = trconf.max_lr * jnp.sqrt(trconf.warmup_steps) / jnp.sqrt(step + 1)
    return jnp.where(step < trconf.warmup_steps, warmup_lr, regular_lr)


# Generate a weight decay mask
# Exclude biases and layer norm /rms norm parameters
graphdef, params, _ = nnx.split(m, nnx.Param, nnx.Variable)
weight_decay_mask = jax.tree.map(
    lambda x: len(x.value.shape) > 1, params, is_leaf=lambda n: isinstance(n, nnx.Param)
)

tx = optax.chain(
    optax.clip_by_global_norm(trconf.max_grad_norm),
    optax.adamw(
        inverse_sqrt_schedule,
        b1=trconf.adam_b1,
        b2=trconf.adam_b2,
        weight_decay=trconf.weight_decay,
        mask=weight_decay_mask,
    ),
)
optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)


# Count the number of weight decay params
def f(x, y):
    if x:
        return y.size
    return 0


weight_decay_params = jax.tree_util.tree_map(f, weight_decay_mask, params)
weight_decay_param_count = jax.tree_util.tree_reduce(
    lambda x, y: x + y, weight_decay_params, 0
)
logger.info(f"Weight decay param count: {weight_decay_param_count:,}")
logger.info(f"Training config:\n{pformat(trconf)}")
logger.info(f"Effective batch size per device: {trconf.mB // num_devices}")
assert trconf.mB * trconf.T == trconf.num_tokens_per_batch

# Set up Dataloader

train_dl = HuggingfaceDataLoader(
    dirpath="datasets/fineweb-edu/fineweb100B",
    batch_size=trconf.mB,
    block_size=trconf.T,
    device_rank=1,
    label="train",
)

val_dl = HuggingfaceDataLoader(
    dirpath="datasets/fineweb-edu/fineweb100B",
    batch_size=trconf.mB,
    block_size=trconf.T,
    device_rank=1,
    label="val",
    quiet=True,
)


# Loss function
def moe_loss_fn(model, batch, targets):
    logits, load_balance_loss, z_loss = model(batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    logit_loss = loss.mean()
    loss = (
        logit_loss
        + model.config.load_balance_loss_coeff * load_balance_loss
        + model.config.z_loss_coeff * z_loss
    )
    return loss, (logit_loss, load_balance_loss, z_loss)


@nnx.jit
def step_fn(model, optimizer, batch, target):
    (loss, (logits_loss, load_balance_loss, z_loss)), grads = nnx.value_and_grad(
        moe_loss_fn, has_aux=True
    )(model, batch, target)
    optimizer.update(model, grads)
    return loss, logits_loss, load_balance_loss, z_loss


@nnx.jit
def compute_val_loss(model, x, y):
    """Compute loss without computing gradients."""
    logits, load_balance_loss, z_loss = model(x)
    logits_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss = (
        logits_loss
        + model.config.load_balance_loss_coeff * load_balance_loss
        + model.config.z_loss_coeff * z_loss
    )
    return loss, logits_loss


def run_validation(model, val_dataloader, data_sharding, num_batches: int = 50):
    """
    Run validation on the model.

    Args:
        model: The model to validate
        val_dataloader: Validation dataloader
        data_sharding: JAX sharding for data
        num_batches: Number of batches to validate on

    Returns:
        Tuple of (avg_loss, avg_logits_loss)
    """
    # Disable noise and aux losses for validation
    model.train(add_noise=False, load_balance_loss=False, z_loss=False)
    total_loss = 0.0
    total_logits_loss = 0.0

    for _ in range(num_batches):
        batch, target = val_dataloader()
        batch = jax.device_put(batch.squeeze(), data_sharding)
        target = jax.device_put(target.squeeze(), data_sharding)
        loss, logits_loss = compute_val_loss(model, batch, target)
        total_loss += loss.item()
        total_logits_loss += logits_loss.item()

    # Re-enable training mode
    model.train(add_noise=False, load_balance_loss=True, z_loss=True)
    return total_loss / num_batches, total_logits_loss / num_batches


# Train

with mesh:
    train_losses = []
    append_to_csv(
        log_dir / f"{run_name}_train.csv",
        [
            "step",
            "lr",
            "loss",
            "load_balance_loss",
            "z_loss",
            "time",
            "tokens_processed",
            "tokens_per_sec",
        ],
    )
    if trconf.val:
        append_to_csv(
            log_dir / f"{run_name}_val.csv",
            ["step", "loss", "logits_loss"],
        )
    logger.info(f"Starting from step: {optimizer.step.value.item()}")
    start = False
    data_sharding = NamedSharding(
        mesh,
        PartitionSpec(
            "devices",
        ),
    )
    m.train(add_noise=False, load_balance_loss=True, z_loss=True)
    try:
        while optimizer.step.value.item() < trconf.max_steps:
            step = optimizer.step.value.item()
            batch, target = train_dl()
            batch = jax.device_put(batch.squeeze(), data_sharding)
            target = jax.device_put(target.squeeze(), data_sharding)
            avg_loss, logits_loss, load_balance_loss, z_loss = step_fn(
                m, optimizer, batch, target
            )
            if step % trconf.print_interval == 0:
                if start is False:
                    start = time.time()
                    iter_time = 0
                    tokens_per_sec = 0
                else:
                    total_time = time.time() - start
                    iter_time = total_time / trconf.print_interval
                    tokens_per_sec = (
                        trconf.print_interval * trconf.mB * trconf.T / total_time
                    )

                tokens_processed = (step + 1) * trconf.mB * trconf.T
                lr = inverse_sqrt_schedule(step)
                avg_loss = avg_loss.item()
                logits_loss = logits_loss.item()
                load_balance_loss = load_balance_loss.item()
                z_loss = z_loss.item()

                train_losses.append((step, avg_loss))
                append_to_csv(
                    log_dir / f"{run_name}_train.csv",
                    [
                        step,
                        lr,
                        avg_loss,
                        load_balance_loss,
                        z_loss,
                        iter_time * 1000,
                        tokens_processed,
                        tokens_per_sec,
                    ],
                )
                logger.info(
                    f"{step} | lr: {lr:0.4f} | "
                    f"loss: {avg_loss:0.4f} | "
                    f"logits loss: {logits_loss:0.4f} | "
                    f"load balance loss: {load_balance_loss:0.4f} | "
                    f"z loss: {z_loss:0.4f} | "
                    f"avg iter time: {iter_time*1000:0.2f}ms | "
                    f"avg tok/sec: {tokens_per_sec:,.2f} | "
                    f"tokens processed: {tokens_processed:,}"
                )
                start = time.time()
            if trconf.val and step > 0 and step % trconf.val_interval == 0:
                logger.info(f"Running validation at step {step}...")
                val_loss, val_logits_loss = run_validation(
                    m, val_dl, data_sharding, num_batches=trconf.val_batches
                )
                logger.info(
                    f"Validation | step: {step} | loss: {val_loss:.4f} | "
                    f"logits loss: {val_logits_loss:.4f}"
                )
                append_to_csv(
                    log_dir / f"{run_name}_val.csv",
                    [step, val_loss, val_logits_loss],
                )
            if (
                trconf.checkpoint_model
                and step > 0
                and step % trconf.checkpoint_interval == 0
            ):
                logger.info(f"Saving model checkpoint at step {step}")
                save_checkpoint(m, output_dir, run_name, step)
            if (
                trconf.checkpoint_optimizer
                and step > 0
                and step % trconf.checkpoint_interval == 0
            ):
                logger.info(f"Saving optimizer checkpoint at step {step}")
                save_optimizer_state(output_dir, run_name, config, optimizer)
    except KeyboardInterrupt:
        logger.warning("Received KeyboardInterrupt. Exiting...")
    finally:
        plt.figure(figsize=(7, 5))
        plt.plot(
            [x[0] for x in train_losses],
            [x[1] for x in train_losses],
            label="train loss",
        )
        plt.yticks(ticks=np.arange(0, 12, 0.5))
        plt.grid()
        plt.legend()
        plt.savefig(
            log_dir / f"{run_name}_train.png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        if trconf.checkpoint_model:
            save_checkpoint(m, output_dir, run_name, optimizer.step.value.item())
        if trconf.checkpoint_optimizer:
            save_optimizer_state(output_dir, run_name, config, optimizer)
        logger.info("Training completed.")
