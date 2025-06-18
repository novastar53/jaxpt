from functools import partial

import jax
import flax.nnx as nnx
import jax.numpy as jnp

from jaxpt.modules.config import Config

class MOE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden * config.n_experts,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),(None, "model")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("model")),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_mlp_hidden * config.n_experts,
            config.n_embed,
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ),(None, "model")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("model",)),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        match config.glu_activation:
            case "sigmoid":
                self.activation = nnx.sigmoid
            case "gelu":
                self.activation = partial(nnx.gelu, approximate=True)
            case "swish" | "silu":
                self.activation = nnx.silu
            case _:
                self.activation = nnx.sigmoid

        self.router_gate = nnx.Linear(
            config.n_embed, 
            config.n_experts, 
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(
                stddev=0.02), (None, "model")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("model",)),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.config = config
    
    def __call__(self, x):
        B, T, C = x.shape
        H = self.config.n_mlp_hidden
        nE = self.config.n_experts
        K = self.config.n_top_k_experts

        router_logits = self.router_gate(x) # B x T x nE
        router_probs = nnx.softmax(router_logits, axis=-1) # B x T x nE
        router_top_k_probs, router_top_k_indices  = jax.lax.top_k(router_probs, K)
        router_top_k_total_probs = jnp.sum(router_top_k_probs, axis=-1)
        router_top_k_probs /= router_top_k_total_probs[..., None]

        c_fc_kernel = self.c_fc.kernel.reshape(C, H, nE) # (C x H x nE)
        c_fc_top_k = jnp.take(c_fc_kernel, router_top_k_indices, axis=-1)
        h = jnp.einsum("btc,chbtk->bthk", x, c_fc_top_k) # (B, T, H, K)

        h = nnx.gelu(h, approximate=True)

        c_proj_kernel = self.c_proj.kernel.reshape(H, C, nE)
        c_proj_top_k = jnp.take(c_proj_kernel, router_top_k_indices, axis=-1)
        o = jnp.einsum("bthk,hcbtk->btck", h, c_proj_top_k)
        o = jnp.einsum("btck,btk->btc", o, router_top_k_probs) # weighted sum of experts
        return o


class GLU(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                (None, "model")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("model,")),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.gate = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                (None, "model")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("model",)),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_mlp_hidden,
            config.n_embed,
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ), (None, "model")),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros, ("model",)),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        match config.glu_activation:
            case "sigmoid":
                self.activation = nnx.sigmoid
            case "gelu":
                self.activation = partial(nnx.gelu, approximate=True)
            case "swish" | "silu":
                self.activation = nnx.silu
            case _:
                self.activation = nnx.sigmoid

    def __call__(self, x):
        h = self.c_fc(x)
        g = self.gate(x)
        g = self.activation(g)
        h = g * h
        y = self.c_proj(h)
        return y


class MLP(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                (None, "model")),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                ("model",)),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_mlp_hidden,
            config.n_embed,
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ), (None, "model")),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros, ("model",)),
            dtype=config.dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.c_fc(x)
        x = nnx.gelu(x, approximate=True)
        x = self.c_proj(x)
        return x
