#!/usr/bin/env python
# coding: utf-8

# # Let's Train Tiny MoE

# ### Configure the machine and install packages



from typing import Literal

import os

#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax

import logging
from pprint import pformat


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

platform : Literal["darwin", "colab", "cuda", "tpu"] = "darwin"

devices = jax.devices()
if any(d.platform == "gpu" for d in devices):
    platform = "cuda"
if any(d.platform == "tpu" for d in devices):
    platform = "tpu"

logger.info(f"Running on {platform}")



from pathlib import Path
import sys

jaxpt_dir = str(Path().absolute().parent / "src" )

sys.path.append(jaxpt_dir)


logger.info(f"Jaxpt dir {jaxpt_dir}")




import os


import warnings

import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import flax
import jax.numpy as jnp
import numpy as np

# Hardware setup
logger.info(f"JAX version: {jax.__version__}")
logger.info(f"Flax version: {flax.__version__}")

devices = jax.devices()
num_devices = len(devices)
logger.info(f"Available devices: {num_devices}")

requested_device = "gpu"

jax.config.update("jax_platform_name", requested_device) # Make sure we're using the GPU

device = jax.default_backend()
if device != requested_device:
    warnings.warn(f"not using {requested_device}. Using {device}")
else:
    logger.info(f"using {device}")


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

jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32") # Set the default precision for matrix multiplication

#jax.config.update("jax_enable_x64", True) # Make sure the highest precision is enabled in case we need
#os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
#os.environ["JAX_ENABLE_X64"] = "False"

if device == "tpu":
    def list_tpu_memory():
        devices = jax.devices()
        for device in devices:
            if 'TPU' in str(device.device_kind):
                print(f"Device: {device}, Memory: {device.memory_stats()['bytes_limit']/(1024*1024)},  Used: {device.memory_stats()['bytes_in_use']/(1024*1024)}")

    list_tpu_memory()

# Set up device mesh


mesh = jax.sharding.Mesh(jax.devices(), ["devices"])

# ### Initialize the model and perform a sanity check

from datetime import datetime

from jaxpt.checkpointers import save_checkpoint, load_checkpoint, load_checkpoint_from_gcloud
from jaxpt.utils import generate_readable_code

if platform == "cuda":
  output_dir = Path("/workspace/alpha_training_runs") # Lambda Labs setup
else:
  output_dir = Path().absolute().parent  / "alpha_training_runs" # Local setup
logger.info(f"Output dir: {output_dir}")

timestamp = datetime.now().strftime("%Y%m%d")
random_code = generate_readable_code()

run_dirname = f"run_{timestamp}_{random_code}"
logger.info(f"Run: {run_dirname}")

from flax import nnx

from pprint import pprint

from jaxpt.models import Tiny_MoE_Config, Tiny_MoE
from jaxpt.utils import count_params, create_sharded_model

import tiktoken
from transformers import AutoTokenizer

default = jax.random.key(1337)
gate_noise = jax.random.key(42)
rngs = nnx.Rngs(default=default, gate_noise=gate_noise)

config = Tiny_MoE_Config(
                     name="Tiny_MoE",
                     dtype=jnp.bfloat16, \
                     vocab_size=49152,
                     n_layer=30,
                     block_size=2048,
                     n_head=9,
                     n_kv_head=3,
                     n_embed=576,
                     n_mlp_hidden=1536,
                     moe_bias=False,
                     mlp_bias=False,
                     attention_bias=False,
                     z_loss_coeff=1e-4,
                     expert_weight_priority=False,
                     load_factor=1.25,
                     ln_epsilon = 1e-5,
                     sdpa_implementation="cudnn" if device=="gpu" else "xla")
pprint(config)

with mesh:
    m = create_sharded_model(Tiny_MoE, config, rngs)
    #m = load_checkpoint(Tiny_MoE, output_dir, config, "run_20250726_excudate_quilling", 2680, rngs)
    #m = load_checkpoint_from_gcloud(Tiny_MoE, config, output_dir, "alpha_training_runs", "run_20250728_mercapto_inkstand", "120000", rngs)
    #m = from_hf_pretrained(config, rngs)

    graphdef, rngstate, state = nnx.split(m, nnx.RngState, ...)
    total_params = count_params(m)
    moe_params = count_params(m, "moe")

    logger.info(f"Parameter Count: {total_params:,}")
    logger.info(f"MOE Parameter Count: {moe_params:,}")
    logger.info(f"Replicated Parameter Count: {total_params - moe_params:,}")

  
# ### Configure Training Run



import orbax.checkpoint as ocp

# Set up save and load optimizer

def save_optimizer_state(m, optimizer):
  state_dirpath = (
    output_dir / m.config.name / "optimizer_checkpoints" / run_dirname
  )
  state_dirpath.mkdir(parents=True, exist_ok=True)
  _, state = nnx.split(optimizer)
  state.model = None
  cp = ocp.StandardCheckpointer()
  logger.info(f"Saving optimizer state to {state_dirpath}/step-{optimizer.step.value.item()}")
  cp.save(state_dirpath / f"step-{optimizer.step.value.item()}", state)
  cp.wait_until_finished()

def load_optimizer_state(model, optimizer, run_dirname, step):
  cp = ocp.StandardCheckpointer()
  graphdef, state = nnx.split(optimizer)
  state_model = state.model
  state.model = None
  state = cp.restore(output_dir / optimizer.model.config.name / "optimizer_checkpoints" / run_dirname / f"step-{step}", target=state)
  state.model = state_model
  optimizer = nnx.merge(graphdef, state)
  return optimizer




import dataclasses

import optax

############################
# Nvidia A100 (x 8) Config
############################

@dataclasses.dataclass
class TrainerConfig:
  num_tokens: int = int(236e9)
  num_tokens_per_batch: int = 2**19 # 2**19, 0.5 million as per the GPT 3.5 paper
  mB: int = 32 * num_devices
  T: int = 2048
  max_steps: int = int(num_tokens // num_tokens_per_batch)
  max_lr: float = 6e-4
  min_lr: float = max_lr * 0.1
  max_grad_norm: float = 1.0  # Clip gradients to this norm
  warmup_steps: int = max_steps // 100
  print_interval: int = 100
  eval_interval: int = 5000
  checkpoint_interval: int = 10000
  grad_accumulation_steps: int = num_tokens_per_batch // (mB * T) # Number of steps over which to average the gradient


##############
# CPU Config #
##############

trconf = TrainerConfig()


# Set up the optimizer
def trapezoidal_schedule(step):

    warmup_lr = trconf.max_lr * (step + 1) / trconf.warmup_steps
    cooldown_lr = trconf.max_lr * (trconf.max_steps - step - 1) / (0.2 * trconf.max_steps)

    return jnp.where(step < trconf.warmup_steps,
                     warmup_lr,
                     jnp.where(step < 0.8 * trconf.max_steps, trconf.max_lr, cooldown_lr))

# Generate a weight decay mask
# First split the model into params and variables
graphdef, params, variables = nnx.split(m, nnx.Param, nnx.Variable)
# Then create a mask for the weight decay params
weight_decay_mask = jax.tree.map(lambda x: len(x.value.shape) > 1, params, is_leaf=lambda n: isinstance(n, nnx.Param))

tx = optax.chain(
    #optax.clip_by_global_norm(trconf.max_grad_norm),
    optax.adamw(trapezoidal_schedule, b1=0.9, b2=0.95, weight_decay=0.1, mask=weight_decay_mask),
)
optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)

#optimizer = load_optimizer_state(m, optimizer, "run_20250726_excudate_quilling", 2680)

# count the number of weight decay params
def f(x, y):
    if x:
        return y.size
    return 0

weight_decay_params = jax.tree_util.tree_map(f, weight_decay_mask, params)
weight_decay_param_count = jax.tree_util.tree_reduce(lambda x, y: x + y, weight_decay_params, 0)
logger.info(f"weight decay param count: {weight_decay_param_count:,}")
logger.info("Model Config:\n%s", pformat(trconf))
logger.info(f"effective batch size: {trconf.grad_accumulation_steps * trconf.mB}")
logger.info(f"effective batch size per device: {trconf.grad_accumulation_steps * trconf.mB // num_devices}")
assert(trconf.grad_accumulation_steps == 1)

# ### DataLoader and Validation Setup
# 
# 

import os

from jaxpt.dataloaders import BlendedCloudDataLoader, CloudDataLoader

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./alpha-448101-282bc1b884cd.json"
'''
train_dl = CloudDataLoader(bucket_name="jaxpt_datasets",
                      bucket_prefix="fineweb-edu-100b/processed",
                      batch_size=trconf.mB,
                      block_size=trconf.T,
                      device_rank=1,
                      label="train")
'''


train_dl = BlendedCloudDataLoader(
    device_rank=1,
    block_size=trconf.T,
    batch_size=trconf.mB,
    bucket_names=["jaxpt_datasets", "jaxpt_datasets", "jaxpt_datasets"],
    bucket_prefixes=["smollm-corpus/processed/fineweb-edu-dedup",
    "smollm-corpus/processed/python-edu",
    "smollm-corpus/processed/cosmopedia-v2"],
    proportions=[85, 1, 12],
    label="train"
)




from jaxpt.utils import append_to_csv

# Create log dir
log_dir = output_dir / m.config.name / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Log directory: {log_dir}")

train_losses = []
append_to_csv(log_dir / f"{run_dirname}_train.csv", ["step", "lr", "loss", "load_balance_loss", "z_loss", "time", "tokens_processed", "tokens_per_sec"])
logger.info(f"Starting from step: {optimizer.step.value.item()}")
start = False


import time

import matplotlib.pyplot as plt

def moe_loss_fn(model, batch, targets):
    logits, load_balance_loss, z_loss = model(batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    loss = loss.mean() 
    loss += model.config.load_balance_loss_coeff * load_balance_loss
    loss += model.config.z_loss_coeff * z_loss
    return loss, (load_balance_loss, z_loss)


@nnx.jit
def train_step(model, optimizer, batch, target):
    (loss, (load_balance_loss, z_loss)), grads = nnx.value_and_grad(moe_loss_fn, has_aux=True)(model, batch, target)
    optimizer.update(model, grads)
    return loss, load_balance_loss, z_loss


with mesh:
  data_sharding = NamedSharding(mesh, PartitionSpec("devices",))
  m.train(add_noise=True, load_balance_loss=True, z_loss=True)
  try:
    while optimizer.step.value.item() < trconf.max_steps:
      step = optimizer.step.value.item()
      batch, target = train_dl()
      batch = jax.device_put(batch.squeeze(), data_sharding)
      target = jax.device_put(target.squeeze(), data_sharding)
      avg_loss, load_balance_loss, z_loss = train_step(m, optimizer, batch, target)
      if step % trconf.print_interval == 0:
        if not start:
          start = time.time()
          iter_time = 0
          tokens_per_sec = 0
        else:
          total_time = (time.time() - start)
          iter_time =  total_time / trconf.print_interval
          tokens_per_sec = trconf.print_interval * trconf.mB * trconf.T * trconf.grad_accumulation_steps / total_time

        tokens_processed = (step+1) * trconf.grad_accumulation_steps * trconf.mB * trconf.T
        lr = trapezoidal_schedule(step)
        avg_loss = avg_loss.item()

        train_losses.append((step, avg_loss))
        append_to_csv(log_dir / f"{run_dirname}_train.csv", [step, lr.item(), avg_loss, load_balance_loss.item(), z_loss.item(), iter_time*1000, tokens_processed, tokens_per_sec])
        logger.info(f"{step} | lr: {lr:0.4f} | "
                    f"loss: {avg_loss:0.4f} | "
                    f"load_balance_loss: {load_balance_loss.item():0.4f} | "
                    f"z_loss: {z_loss.item():0.4f} | "
                    f"time: {iter_time*1000:0.2f}ms | "
                    f"tokens processed: {tokens_processed:,} | "
                    f"tok/sec: {tokens_per_sec:,.2f}")
        start = time.time()
      if step > 0 and step % trconf.eval_interval == 0:
        logger.info("Evaluation TBD")
      if step > 0 and step % trconf.checkpoint_interval == 0:
        logger.info(f"Saving checkpoint at step {step}")
        #save_checkpoint(m, output_dir, run_dirname, step)
        #save_optimizer_state(optimizer)
  except KeyboardInterrupt:
      logger.info("Received KeyboardInterrupt. Exiting...")
  finally:
    #plt.figure(figsize=(7, 5))
    #plt.plot([x[0] for x in train_losses], [x[1] for x in train_losses], label="train loss")
    #plt.yticks(ticks=np.arange(0, 12, 0.5))
    #plt.grid()
    #plt.legend()
    #plt.savefig(log_dir / f"{run_dirname}.png", dpi=300, bbox_inches="tight", transparent=True)
#
#    save_checkpoint(m, output_dir, run_dirname, optimizer.step.value.item())
#    save_optimizer_state(m, optimizer)
    logger.info("Done.")
