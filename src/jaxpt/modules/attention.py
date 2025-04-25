import math

import jax
import flax.nnx as nnx
import jax.numpy as jnp

from jaxpt.modules import Config


def calc_vanilla_attn(q, k, v, mask):
    B, T, n_head, hs = q.shape
    q = jnp.transpose(q, axes=(0, 2, 1, 3)) # (B, n_head, T, hs)
    k = jnp.transpose(k, axes=(0, 2, 3, 1)) # (B, n_head, hs, T)
    v = jnp.transpose(v, axes=(0, 2, 1, 3)) # (B, n_head, T, hs)
    att = ( q @ k ) / math.sqrt(k.shape[-1])  # B, n_head, T, T
    att = jnp.where(mask[:, :, :T, :T] == 0.0, float('-inf'), att)
    att = jax.nn.softmax(att, axis=-1)
    y = att @ v # (B, n_head, T, T) x (b, n_head, T, hs) -> (B, n_head, T, hs)
    y = jnp.transpose(y, axes=(0, 2, 1, 3)) # (B, T, n_head, hs)
    return y, att


class CausalSelfAttention(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
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
        self.use_vanilla_attn = config.use_vanilla_attn

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

        if self.use_vanilla_attn:
            y, _ = calc_vanilla_attn(q, k, v, self.mask)
        else:
            y = jax.nn.dot_product_attention(
                q, k, v, is_causal=True, implementation=self.implementation
            )

        y = jnp.reshape(y, (B, T, C))  # (B, T, C)
        y = self.c_proj(y)
        return y

class RoPEAttention(nnx.Module):
    def __init__(self, config: Config, wpe: nnx.Param, rngs: nnx.Rngs):
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
        self.wpe = wpe

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
        self.use_vanilla_attn = config.use_vanilla_attn

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)  # B, T, 3 * C
        q, k, v = jnp.split(qkv, 3, axis=-1)  # 3 * (B, T, C)

        pos = jnp.arange(0, T, dtype=jnp.uint16)
        pos_emb = self.wpe(pos)

        q = jnp.reshape(
            q, (B, self.n_head, T, C // self.n_head)
        )  
        q += pos_emb
        q = jnp.reshape(
            q, (B, T, self.n_head, C // self.n_head)
        )

        k = jnp.reshape(
            k, (B, self.n_head, T, C // self.n_head)
        )  
        k += pos_emb
        k = jnp.reshape(
            k, (B, T, self.n_head, C // self.n_head)
        )

        v = jnp.reshape(
            v, (B, T, self.n_head, C // self.n_head)
        )  # B, T, n_head, C // n_head

        if self.use_vanilla_attn:
            y, _ = calc_vanilla_attn(q, k, v, self.mask)
        else:
            y = jax.nn.dot_product_attention(
                q, k, v, is_causal=True, implementation=self.implementation
            )

        y = jnp.reshape(y, (B, T, C))  # (B, T, C)
        y = self.c_proj(y)
        return y

