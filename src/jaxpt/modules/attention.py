from typing import Literal
import abc
import math

import jax
import flax.nnx as nnx
import jax.numpy as jnp

from jaxpt.modules.config import Config
from jaxpt.modules.position import RoPE_Llama


def _calc_slow_attn(q, k, v, mask, bias=None):
    _, T, _, _ = q.shape

    if bias is None:
        bias = jnp.zeros(shape=(T, T), dtype=q.dtype)

    q = jnp.transpose(q, axes=(0, 2, 1, 3))  # (B, n_head, T, hs)
    k = jnp.transpose(k, axes=(0, 2, 3, 1))  # (B, n_head, hs, T)
    v = jnp.transpose(v, axes=(0, 2, 1, 3))  # (B, n_head, T, hs)
    att = (q @ k) + bias
    att = att / math.sqrt(k.shape[-1])  # B, n_head, T, T
    att = jnp.where(mask[:, :, :T, :T] == 0.0, float("-inf"), att)
    att = jax.nn.softmax(att, axis=-1)
    y = att @ v  # (B, n_head, T, T) x (b, n_head, T, hs) -> (B, n_head, T, hs)
    y = jnp.transpose(y, axes=(0, 2, 1, 3))  # (B, T, n_head, hs)
    return y, att


def _calc_attention(
    query,
    key,
    value,
    mask=None,
    bias=None,
    implementation: Literal["xla", "cudnn", "slow"] | None = None,
):
    output_shape = jnp.asarray(query).shape

    match implementation:
        case "xla" | "cudnn":
            if mask is not None:
                # Convert a (B x T) mask to a (1 x B x T x T) mask
                mask1 = mask[..., :, None]
                mask2 = mask[..., None, :]
                mask = mask1 & mask2
                mask = mask[:, None, :, :]
            out = jax.nn.dot_product_attention(
                query,
                key,
                value,
                mask=mask,
                bias=bias,
                is_causal=True,
                implementation=implementation,
            )
        case _:
            out, _ = _calc_slow_attn(query, key, value, mask, bias)

    return jnp.reshape(out, output_shape)


class SelfAttentionBase(nnx.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, x):
        raise NotImplementedError


class CausalSelfAttention(SelfAttentionBase):
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
        if config.sdpa_implementation == "slow":
            self.mask = nnx.Variable(
                jnp.tril(
                    jnp.ones(
                        (config.block_size, config.block_size),
                        dtype=config.dtype,
                    ).reshape((1, 1, config.block_size, config.block_size))
                )
            )
        self.n_head = config.n_head
        self.n_embed = config.n_embed
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

        y = _calc_attention(q, k, v, implementation=self.implementation)

        y = jnp.reshape(y, (B, T, C))  # (B, T, C)
        y = self.c_proj(y)
        return y


class GQ_Attention(SelfAttentionBase, RoPE_Llama):
    def __init__(
        self, config: Config, rope_omega: nnx.Variable, rngs: nnx.Rngs
    ):
        RoPE_Llama.__init__(self, omega=rope_omega)

        self.config = config

        self.wq = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=config.init_stddev),
            bias_init=nnx.initializers.zeros,
            use_bias=config.attention_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.wkv = nnx.Linear(
            config.n_embed,
            2 * config.n_kv_head * config.n_embed // config.n_head,
            kernel_init=nnx.initializers.normal(stddev=config.init_stddev),
            bias_init=nnx.initializers.zeros,
            use_bias=config.attention_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.wproj = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init=nnx.initializers.normal(
                stddev=config.init_stddev * (2 * config.n_layer) ** -0.5
            ),
            bias_init=nnx.initializers.zeros,
            use_bias=config.attention_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.implementation = config.sdpa_implementation
        self.key_cache = None
        self.value_cache = None

    def __call__(self, x, mask=None):
        B, x_T, C = x.shape
        q = self.wq(x)  # (B, x_T, C)
        kv = self.wkv(x)  # (B, x_T, 2 * n_kv_head * C // n_head)
        k, v = jnp.split(
            kv, 2, axis=-1
        )  # 2 x (B, x_T, n_kv_head * C // n_head)

        k_T, v_T = x_T, x_T

        q = q.reshape((B, x_T, self.n_head, C // self.n_head))
        offset = 0
        if self.key_cache is not None:
            offset = self.key_cache.shape[1]
        q = self.apply_rope(q, offset=offset)

        if self.config.use_cache is True:
            if self.key_cache is None:
                self.key_cache = k
                k_T = x_T
            else:
                self.key_cache = jnp.concat((self.key_cache, k), axis=1)
                self.key_cache = self.key_cache[:, -self.config.block_size :, :]
                k_T = self.key_cache.shape[1]
                k = self.key_cache

            if self.value_cache is None:
                self.value_cache = v
                v_T = x_T
            else:
                self.value_cache = jnp.concat((self.value_cache, v), axis=1)
                self.value_cache = self.value_cache[
                    :, -self.config.block_size :, :
                ]
                v_T = self.key_cache.shape[1]
                v = self.value_cache

        k = k.reshape((B, k_T, self.n_kv_head, C // self.n_head))
        k = self.apply_rope(k)

        v = v.reshape((B, v_T, self.n_kv_head, C // self.n_head))

        y = _calc_attention(
            q, k, v, implementation=self.implementation, mask=mask
        )  # (B, T, n_head, C // n_head)

        y = jnp.reshape(y, (B, x_T, C))
        y = self.wproj(y)
        return y
