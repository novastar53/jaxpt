from typing import Literal

import os

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'

import jax

platform : Literal["darwin", "colab", "cuda", "tpu"] = "darwin"

from pathlib import Path
import sys

jaxpt_dir = str(Path().absolute().parent / "src" )

sys.path.append(jaxpt_dir)
print(jaxpt_dir)
from functools import partial
from dataclasses import dataclass
import random

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding, Mesh
from jax.debug import visualize_array_sharding as viz

import flax.nnx as nnx
import optax

from jaxpt.modules.config import Config
#from jaxpt.utils import create_sharded_model


devices = jax.devices()
print(devices)

mesh = Mesh(devices, ("devices"))
spec = PartitionSpec("devices",)
sharding = NamedSharding(mesh, spec)

@nnx.jit(static_argnums=(0, 1)) #, out_shardings=sharding)
def create_sharded_model(Model, config, rngs):
    model = Model(config=config, rngs=rngs)
    graphdef, state = nnx.split(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = nnx.with_sharding_constraint(
        state, pspecs, mesh=config.mesh
        )
    nnx.update(model, sharded_state)
    return model



@dataclass(unsafe_hash=True)
class MOE_Config(Config):
    n_layer = 1
    top_k = 2
    load_factor = 1.00
    n_experts = len(devices)
    n_embed = 3 
    n_mlp_hidden = 6
    mlp_bias = True
    dtype = jax.numpy.float32
    mesh = mesh

config = MOE_Config()


class Experts(nnx.Module):
    def __init__(self, config, rngs):
        w_c_fc_init = nnx.with_partitioning(
            nnx.initializers.normal(stddev=0.02),
            sharding=("devices",))
        
        b_init = nnx.with_partitioning(
            nnx.initializers.zeros,
            sharding=("devices",))
        
        w_c_proj_init = nnx.with_partitioning(
            nnx.initializers.normal(stddev=0.02 * (2 * config.n_layer) ** -0.5),
            sharding=("devices",)
        )

        self.w_c_fc = nnx.Param(w_c_fc_init(rngs.default(),
            (
                config.n_experts,
                config.n_embed,
                config.n_embed
            )
        ))
        self.b_c_fc = nnx.Param(b_init(rngs.default(),
        (
            config.n_experts,
            1,
            config.n_mlp_hidden
        )))

        self.w_gate = nnx.Param(w_c_fc_init(rngs.default(),
        (
            config.n_experts,
            config.n_embed,
            config.n_mlp_hidden
        )))
        self.b_gate = nnx.Param(b_init(rngs.default(),
        (
            config.n_experts,
            1,
            config.n_mlp_hidden
        )))

        self.w_c_proj = nnx.Param(
            w_c_proj_init(
                rngs.default(),
                (
                    config.n_experts,
                    config.n_mlp_hidden,
                    config.n_embed
                ))
        )
        self.b_c_proj = nnx.Param(
            b_init(
                rngs.default(),
                (
                    config.n_experts,
                    1,
                    config.n_embed
                )
            )
        )

    def __call__(self, x):
        x = jax.lax.with_sharding_constraint(x, spec)
        o = jnp.einsum('eti,eio->eto', x, self.w_c_fc)
        #o = x @ self.w_c_fc
        #h = jnp.einsum('eti,eih->eth', x, self.w_c_fc) + self.b_c_fc
        #g = jnp.einsum('eti,eih->eth', x, self.w_gate) + self.b_gate
        #g = nnx.silu(g)
        #og = jnp.einsum('eth,eth->eth', h, g)
        #o = jnp.einsum('eth,eho->eto', og, self.w_c_proj) + self.b_c_proj
        o = jax.lax.with_sharding_constraint(o, spec)
        return o


class MOE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.router_gate = nnx.Linear(
            config.n_embed,
            config.n_experts,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                sharding=(None,)),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros,
            sharding=(None,)),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.experts = Experts(config, rngs)
        self.top_k = config.top_k
        self.n_experts = config.n_experts
        self.load_factor = config.load_factor
        self.add_noise = False
        self.rngs = rngs
    
    def _get_expert_inputs(self, x, logits):
        T, _ = logits.shape
        _, C = x.shape
        top_k_logits, expert_indices = jax.lax.top_k(logits, self.top_k) # T, top_K
        zeros = jnp.full_like(logits, float('-inf')) # T, n_experts
        expert_probs = jax.nn.softmax(top_k_logits, axis=-1) # T, n_experts  

        # Swap the sequence (T) and top_k dimensions so that when the array is
        # flattened, the higher ranked experts appear first.
        expert_indices = jnp.swapaxes(expert_indices, 0, 1).ravel() # top_K * T
        # Calculate the expert buffer positions all the tokens in the batch
        expert_one_hot = jax.nn.one_hot(expert_indices, self.n_experts, dtype=jnp.int32) # top_K * T, n_experts
        expert_positions = (jnp.cumsum(expert_one_hot, axis=0) * expert_one_hot) - 1 # top_K * T, n_experts
        # Reshape the buffer index to match the original ordering and dimensions
        expert_positions = expert_positions.reshape(-1, T, self.n_experts) # top_K, T, n_experts
        expert_positions = jnp.swapaxes(expert_positions, 0, 1) # T, top_K, n_experts
        # Extract the buffer positions for each token and k, 
        expert_positions = jnp.max(expert_positions, axis=2) # T, top_K
        # Restore the shape and order of expert_indices
        expert_indices = jnp.swapaxes(expert_indices.reshape(-1, T), 0, 1) # T, top_K

        expert_capacity = self.top_k * T 
        zeros = jnp.zeros((self.n_experts, expert_capacity, C)) # n_experts, expert_cap, C

        x = jnp.repeat(x, self.top_k, axis=0)
        expert_inputs = zeros.at[expert_indices.ravel(), 
                                 expert_positions.ravel()].add(x)

        return expert_probs, expert_positions, expert_indices, expert_inputs


    def __call__(self, x):
        B, T, C = x.shape
        logits = self.router_gate(x) # B, T, n_experts
        #if self.add_noise:
        #    logits += 0.01 * jax.random.normal(key=self.rngs.gate_noise(), shape=logits.shape)

        (top_k_probs, 
         expert_positions, 
         expert_indices, 
         expert_inputs) = jax.vmap(
            lambda x, l: self._get_expert_inputs(x, l))(x, logits) # B, n_experts, expert_cap, C 
        
        top_k_probs = jax.lax.with_sharding_constraint(top_k_probs, spec)
        expert_positions = jax.lax.with_sharding_constraint(expert_positions, spec)
        expert_indices = jax.lax.with_sharding_constraint(expert_indices, spec)
        expert_inputs = jax.lax.with_sharding_constraint(expert_inputs, spec)

        expert_capacity = self.top_k * T
        expert_inputs = expert_inputs.reshape(self.n_experts, -1, self.n_experts, expert_capacity, C) # n_experts, batch_per_expert, n_experts, expert_cap, C
        expert_inputs = jnp.swapaxes(expert_inputs, 0, 2) # n_experts, batch_per_expert, n_experts, expert_cap, C
        expert_inputs = expert_inputs.reshape(-1, C) # B * n_experts * expert_cap, C
        expert_inputs = jax.lax.with_sharding_constraint(expert_inputs, spec)
        expert_inputs = expert_inputs.reshape(self.n_experts, B * expert_capacity, C) # n_experts, B * expert_cap, C
        expert_inputs = jax.lax.with_sharding_constraint(expert_inputs, spec)

        #f = input_counters / (B * T)
        #P = jnp.mean(expert_probs, axis=0)
        #aux_loss = jnp.sum(f * P) / (self.n_experts ** 2)

        expert_outputs = self.experts(expert_inputs) # n_experts, B * expert_cap, C
        #expert_outputs = expert_inputs @ jnp.eye(C, C)
        expert_outputs = expert_outputs.reshape(self.n_experts, -1, self.n_experts, expert_capacity, C) # n_experts, batch_per_expert, n_experts, expert_cap, C
        expert_outputs = jnp.swapaxes(expert_outputs, 0, 2) # n_experts, batch_per_expert, n_experts, expert_cap, C
        expert_outputs = expert_outputs.reshape(B, self.n_experts, expert_capacity, C) # B, n_experts, expert_cap, C
        expert_outputs = jax.lax.with_sharding_constraint(expert_outputs, spec)

        expert_outputs = jax.vmap(
            lambda x, i, p: x[i, p]
            )(expert_outputs, expert_indices, expert_positions)

        y_pred = jnp.einsum("BTKC,BTK->BTC", expert_outputs, top_k_probs)       
        y_pred = jax.lax.with_sharding_constraint(y_pred, spec)
        return y_pred, 0, (top_k_probs,)

def loss_fn(model, x, y):
    y_pred, aux_loss, debug_outputs = model(x)
    loss = jnp.mean((y - y_pred)**2) + 0.01 * aux_loss
    return loss, debug_outputs

@nnx.jit
def step(state, x, y):
    (loss, debug_outputs), grads = nnx.value_and_grad(loss_fn, has_aux=True)(state.model, x, y)
    state.update(grads)
    return loss, grads, debug_outputs


from time import time

if __name__ == "__main__":
    with mesh:
        D, B, T, C = 1000, len(devices), 5, config.n_embed

        default = jax.random.key(69)
        gate_noise = jax.random.key(42)
        rngs = nnx.Rngs(default=default, gate_noise=gate_noise)
        model = create_sharded_model(MOE, config, rngs)
        model.train(add_noise=True)
        tx = optax.adam(1e-2)
        state = nnx.Optimizer(model, tx)

        x = jax.random.normal(jax.random.key(1000), (D * B * T, C))

        expert_ids = (x[:, 0] > 0).astype(jnp.int32)
        t = [
            jax.random.normal(jax.random.key(2000), (C, C)),
            jax.random.normal(jax.random.key(3000), (C, C)),
        ]
        def transform(xi, eid):
            return jnp.where(eid == 1, xi @ t[0], xi @ t[1])

        y = jax.vmap(lambda xi, ei: transform(xi, ei))(x, expert_ids)

        x = x.reshape(D, B, T, C)
        y = y.reshape(D, B, T, C)

        indices = list(range(D))

        @nnx.jit
        def run_model(x):
            return model(x)
        #with jax.profiler.trace("./tensorboard"):
        for e in range(100):
            for i in indices:
                start = time()
                x_i = jax.device_put(x[i], sharding)
                y_i = jax.device_put(y[i], sharding)
                loss, grads, debug_outputs = step(state, x_i, y_i)
                top_k_probs, = debug_outputs
                if i % 1000 == 0:
                    end = time()
                    iter_time = 1024 * (end - start) / 1000
                    print(f"{e=}, {i=}, {loss.item()=}, {iter_time=:0.4f}")
