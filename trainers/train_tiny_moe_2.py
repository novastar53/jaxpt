#!/usr/bin/env python
# coding: utf-8

# # Let's Train Tiny MoE

# ### Configure the machine and install packages

# In[1]:


from typing import Literal

import os

#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax

platform : Literal["darwin", "colab", "cuda", "tpu"] = "darwin"

devices = jax.devices()
if any(d.platform == "gpu" for d in devices):
    platform = "cuda"
if any(d.platform == "tpu" for d in devices):
    platform = "tpu"

print(f"Running on {platform}")


# In[2]:

from pathlib import Path
import sys

jaxpt_dir = str(Path().absolute().parent / "src" )

sys.path.append(jaxpt_dir)
print(f"Jaxpt dir {jaxpt_dir}")


# In[3]:


import os


import warnings

import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import flax
import jax.numpy as jnp
import numpy as np

# Hardware setup
print("JAX version:", jax.__version__)
print("Flax version", flax.__version__)

devices = jax.devices()
num_devices = len(devices)
print("Available devices:", num_devices)

requested_device = "gpu"

jax.config.update("jax_platform_name", requested_device) # Make sure we're using the GPU

device = jax.default_backend()
if device != requested_device:
    warnings.warn(f"not using {requested_device}. Using {device}")
else:
    print(f"using {device}")


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

# In[4]:


from datetime import datetime
import random
import string

from jaxpt.checkpointers import save_checkpoint, load_checkpoint, load_checkpoint_from_gcloud
from jaxpt.utils import generate_readable_code

if platform == "cuda":
  output_dir = Path("/workspace/alpha_training_runs") # Lambda Labs setup
else:
  output_dir = Path().absolute().parent  / "alpha_training_runs" # Local setup
print(f"Output dir: {output_dir}")

timestamp = datetime.now().strftime("%Y%m%d")
random_code = generate_readable_code()

run_dirname = f"run_{timestamp}_{random_code}"
print(f"Run: {run_dirname}")



# In[5]:


from flax import nnx


# In[6]:

from pprint import pprint

from jaxpt.models import Tiny_MoE_2_Config, Tiny_MoE_2
from jaxpt.utils import count_params, create_sharded_model

import tiktoken
from transformers import AutoTokenizer

default = jax.random.key(1337)
gate_noise = jax.random.key(42)
rngs = nnx.Rngs(default=default, gate_noise=gate_noise)
rngs = nnx.Rngs(0)

config = Tiny_MoE_2_Config(
                     name="Tiny_MoE",
                     dtype=jnp.bfloat16, \
                     vocab_size=50304,
                     n_layer=32,
                     block_size=2048,
                     n_head=12,
                     n_kv_head=4,
                     n_embed=768,
                     n_mlp_hidden=2304,
                     expert_weight_priority=False,
                     load_factor=1.25,
                     sdpa_implementation="cudnn" if device=="gpu" else "xla")
pprint(config)

with mesh:
    m = create_sharded_model(Tiny_MoE_2, config, rngs)

    graphdef, rngstate, state = nnx.split(m, nnx.RngState, ...)
    total_params = count_params(m)
    moe_params = count_params(m, "moe")

    print(f"Parameter Count: {total_params:,}")
    print(f"MOE Parameter Count: {moe_params:,}")
    print(f"Replicated Parameter Count: {total_params - moe_params:,}")

  
# ### Configure Training Run

# In[7]:


import orbax.checkpoint as ocp

# Set up save and load optimizer

def save_optimizer_state(optimizer):
  state_dirpath = (
    output_dir / optimizer.model.config.name / "optimizer_checkpoints" / run_dirname
  )
  state_dirpath.mkdir(parents=True, exist_ok=True)
  _, state = nnx.split(optimizer)
  state.model = None
  cp = ocp.StandardCheckpointer()
  print(f"Saving optimizer state to {state_dirpath}/step-{optimizer.step.value.item()}")
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


# In[8]:


import dataclasses

import optax

############################
# Nvidia A100 (x 8) Config
############################

@dataclasses.dataclass
class TrainerConfig:
  num_tokens: int = int(111790*1000)
  num_tokens_per_batch: int = 2**18 # 2**19, 0.5 million as per the GPT 3.5 paper
  mB: int = 16 * num_devices
  T: int = 2048
  max_steps: int = int(num_tokens // num_tokens_per_batch)
  max_lr: float = 5e-4
  min_lr: float = max_lr * 0.1
  max_grad_norm: float = 1.0  # Clip gradients to this norm
  warmup_steps: int = max_steps // 100
  print_interval: int = 1
  eval_interval: int = 5000
  checkpoint_interval: int = 10000
  grad_accumulation_steps: int = num_tokens_per_batch // (mB * T) # Number of steps over which to average the gradient


##############
# CPU Config #
##############

trconf = TrainerConfig()
'''
trconf = TrainerConfig(
  num_tokens_per_batch=2**9,
  mB=2**4,
  T=2**5,
  max_steps=9*48, # 6 epoch(s)
  max_lr=6e-4,
  min_lr=6e-5,
  max_grad_norm=1.0,
  warmup_steps=10,
  print_interval=1,
  eval_interval=50,
  checkpoint_interval=0,

)

trconf.grad_accumulation_steps =  trconf.num_tokens_per_batch // (trconf.mB * trconf.T * num_devices) # Number of steps over which to average the gradient
'''

# Set up the optimizer
def trapezoidal_schedule(step):

    warmup_lr = trconf.max_lr * (step + 1) / trconf.warmup_steps
    cooldown_lr = trconf.max_lr * (trconf.max_steps - step) / (trconf.max_steps - 0.8 * trconf.max_steps)

    return jnp.where(step < trconf.warmup_steps,
                     warmup_lr,
                     jnp.where(step < 0.8 * trconf.max_steps, trconf.max_lr, cooldown_lr))

# Generate a weight decay mask
# First split the model into params and variables
graphdef, params, variables = nnx.split(m, nnx.Param, nnx.Variable)
# Then create a mask for the weight decay params
weight_decay_mask = jax.tree_util.tree_map(lambda x: len(x.shape) > 1, params)

tx = optax.chain(
    #optax.clip_by_global_norm(trconf.max_grad_norm),
    #optax.adamw(trapezoidal_schedule, b1=0.9, b2=0.95, weight_decay=0.1, mask=weight_decay_mask),
    optax.adam(trapezoidal_schedule)
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
print(f"weight decay param count: {weight_decay_param_count:,}")
pprint(trconf)
print(f"effective batch size: {trconf.grad_accumulation_steps * trconf.mB}")
print(f"effective batch size per device: ", trconf.grad_accumulation_steps * trconf.mB // num_devices)


# ### DataLoader and Validation Setup
# 
# 

# In[9]:


import os

from jaxpt.dataloaders import DataLoader

train_dl = DataLoader(dirpath="datasets/panchatantra-ryder/processed",
                      batch_size=trconf.mB,
                      block_size=trconf.T,
                      device_rank=1,
                      label="train")


# In[10]:


from jaxpt.utils import append_to_csv

# Create log dir
log_dir = output_dir / m.config.name / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
print(f"Log directory: {log_dir}")

train_losses = []
append_to_csv(log_dir / f"{run_dirname}_train.csv", ["step", "lr", "loss", "time", "tokens_processed", "tokens_per_sec"])
print(f"Starting from step: {optimizer.step.value.item()}")
start = False


# In[22]:

import time

import matplotlib.pyplot as plt

def moe_loss_fn(model, batch, targets):
    logits, aux_loss = model(batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    loss = loss.mean() + model.config.aux_loss_coeff * aux_loss
    return loss, aux_loss


@nnx.jit
def train_step(model, optimizer, batch, target):
    batch = batch.squeeze()
    target = target.squeeze()
    (loss, aux_loss), grads = nnx.value_and_grad(moe_loss_fn, has_aux=True)(model, batch, target)
    optimizer.update(model, grads)
    return loss, aux_loss


with mesh:
  data_sharding = NamedSharding(mesh, PartitionSpec("devices",))
  m.train(add_noise=True, aux_loss=True)
  try:
    while optimizer.step.value.item() < trconf.max_steps:
      step = optimizer.step.value.item()
      batch, target = train_dl()
      batch = jax.device_put(batch.squeeze(), data_sharding)
      target = jax.device_put(target.squeeze(), data_sharding)
      avg_loss, aux_loss = train_step(m, optimizer, batch, target)
      iter_time = time.time() - start
      if step % trconf.print_interval == 0:
        if not start:
          start = time.time()
          iter_time = -1
          sub_step_time = -1
          tokens_per_sec = -1
        else:
          iter_time = (time.time() - start) / trconf.print_interval
          sub_step_time = iter_time / trconf.grad_accumulation_steps
          tokens_per_sec = trconf.mB * trconf.T * trconf.grad_accumulation_steps / iter_time

        tokens_processed = (step+1) * trconf.grad_accumulation_steps * trconf.mB * trconf.T
        lr = trapezoidal_schedule(step)
        avg_loss = avg_loss.item()

        train_losses.append((step, avg_loss))
        append_to_csv(log_dir / f"{run_dirname}_train.csv", [step, lr, avg_loss, iter_time*1000, tokens_processed, tokens_per_sec])
        print(f"{step} | lr: {lr:0.4f} | "
              f"loss: {avg_loss:0.4f} | "
              f"aux_loss: {aux_loss:0.4f} | "
              f"time: {iter_time*1000:0.2f}ms | "
              f"tokens processed: {tokens_processed:,} | "
              f"tok/sec: {tokens_per_sec:,.2f}", end="\n")
        start = time.time()
      if step > 0 and step % trconf.eval_interval == 0:
        print("Evaluation TBD")
      if step > 0 and step % trconf.checkpoint_interval == 0:
        print(f"Saving checkpoint at step {step}")
        #save_checkpoint(m, output_dir, run_dirname, step)
        #save_optimizer_state(optimizer)
  except KeyboardInterrupt:
      print("Received KeyboardInterrupt. Exiting...")
  finally:
    #plt.figure(figsize=(7, 5))
    #plt.plot([x[0] for x in train_losses], [x[1] for x in train_losses], label="train loss")
    #plt.yticks(ticks=np.arange(0, 12, 0.5))
    #plt.grid()
    #plt.legend()
    #plt.savefig(log_dir / f"{run_dirname}.png", dpi=300, bbox_inches="tight", transparent=True)

    #save_checkpoint(m, output_dir, run_dirname, optimizer.step.value.item())
    #save_optimizer_state(optimizer)
    print("Done.")
