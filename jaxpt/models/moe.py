from typing import Literal, Optional
from dataclasses import dataclass

import torch
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from jaxpt.utils import update_param, get_param

from transformers import MOE2LMHeadModel
import orbax.checkpoint as ocp


@dataclass
class MOEConfig:
    dtype: jnp.dtype = jnp.float32
    block_size: int = 1024  # sequence length
    vocab_size: int = 50304  # 50257 padded to the nearest multiple of 64
    n_layer: int = 12  # number of attention blocks
    n_experts: int = 8
    n_head: int = 12  # number of attention heads
    n_embed: int = 768  # number token embedding dimensionsa
    ln_epsilon: float = 1e-5
    sdpa_implementation: Literal["xla", "cudnn"] = "xla"


class CausalSelfAttention(nnx.Module):
    def __init__(self, config: MOEConfig, rngs: nnx.Rngs):
        self.c_attn = nnx.Linear(
            config.n_embed,
            3 * config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init=nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.mask = nnx.Variable(
            jnp.tril(
                jnp.ones(
                    (config.block_size, config.block_size), dtype=config.dtype
                ).reshape((1, 1, config.block_size, config.block_size))
            )
        )
        self.implementation = config.sdpa_implementation

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)  # B, T, 3 * C
        q, k, v = jnp.split(qkv, 3, axis=-1)  # 3 * (B, T, C)

        q = jnp.reshape(
            q, (B, T, self.n_head, C // self.n_head)
        )  # B, T, n_head, C // n_head

        k = jnp.reshape(
            k, (B, T, self.n_head, C // self.n_head)
        )  # B, T, n_head, C // n_head

        v = jnp.reshape(
            v, (B, T, self.n_head, C // self.n_head)
        )  # B, T, n_head, C // n_head

        y = jax.nn.dot_product_attention(
            q, k, v, is_causal=True, implementation=self.implementation
        )

        y = jnp.reshape(y, (B, T, C))  # (B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nnx.Module):
    def __init__(self, config: MOEConfig, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embed,
            4 * config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            4 * config.n_embed,
            config.n_embed,
            kernel_init=nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.c_fc(x)
        x = nnx.gelu(x, approximate=True)
        x = self.c_proj(x)
        return x


class Router(nnx.Module):
    def __init__(self, config: MOEConfig, rngs: nnx.Rngs):
        self.router = nnx.Linear(
            config.n_embed,
            8,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.router(x) # (B x T x 8)
        top_scores, top_idxs = jax.lax.top_k(x, 2) # (B x T x 2)
        gates = nnx.softmax(top_scores, dim=-1)



class Block(nnx.Module):
    def __init__(self, config: MOEConfig, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(
            config.n_embed, epsilon=config.ln_epsilon, dtype=config.dtype, rngs=rngs
        )
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(
            config.n_embed, epsilon=config.ln_epsilon, dtype=config.dtype, rngs=rngs
        )
        self.router = Router(config, rngs)
        self.mlps = [MLP(config, rngs=rngs) for _ in range(config.experts)]

    def __call__(self, x):
        x = self.attn(self.ln_1(x)) + x # (B x T x C)
        router_idxs = self.router(x) # (B x T x 2)
        x = self.mlp(self.ln_2(x)) + x # (B x T X C)
        return x


class MOE(nnx.Module):
    def __init__(self, config: MOEConfig, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(
            config.vocab_size,
            config.n_embed,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            dtype=config.dtype,
            param_dtype=config.dtype,
            rngs=rngs,
        )
        self.wpe = nnx.Embed(
            config.block_size,
            config.n_embed,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.h = [Block(config, rngs=rngs) for _ in range(config.n_layer)]
        self.ln_f = nnx.LayerNorm(
            config.n_embed, epsilon=config.ln_epsilon, dtype=config.dtype, rngs=rngs
        )

    def __call__(self, idx):
        T = idx.shape[1]
        pos = jnp.arange(0, T, dtype=jnp.uint16)
        pos_emb = self.wpe(pos)
        tok_emb = self.wte(idx)
        x = tok_emb + pos_emb
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.wte.attend(x)  # (B x T x V)
        return logits


def save_checkpoint(model, fpath: str):
    _, _, other_state = nnx.split(model, nnx.RngState, ...)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(fpath, other_state)


def from_checkpoint(fpath: str, rngs: nnx.Rngs, config=Optional[MOEConfig]):
    config = config if config else MOEConfig()
    model = MOE(config=config, rngs=rngs)
    _, _, other_state = nnx.split(model, nnx.RngState, ...)
    checkpointer = ocp.StandardCheckpointer()
    other_state = checkpointer.restore(fpath, target=other_state)
    nnx.update(model, other_state)
    return model


