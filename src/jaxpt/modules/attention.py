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
                # Convert a (B x T) mask to a (B x 1 x T x T) mask
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
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                getattr(config, "attn_c_attn_kernel_sharding", (None, "model")),
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                getattr(config, "attn_c_attn_bias_sharding", ("model",)),
            ),
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(
                    stddev=0.02 * (2 * config.n_layer) ** -0.5
                ),
                getattr(config, "attn_c_proj_kernel_sharding", (None, "model")),
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                getattr(config, "attn_c_proj_bias_sharding", ("model",)),
            ),
            dtype=config.dtype,
            param_dtype=config.param_dtype,
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

    def _apply_attn(self, q, k, v, mask=None):
        return _calc_attention(q, k, v, implementation=self.implementation, mask=mask)

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.c_attn(x)  # B, T, 3 * C
        q, k, v = jnp.split(qkv, 3, axis=-1)  # 3 * (B, T, C)

        q = jnp.reshape(q, (B, T, self.n_head, C // self.n_head))  # B, T, n_head, C // n_head

        k = jnp.reshape(k, (B, T, self.n_head, C // self.n_head))  # B, T, n_head, C // n_head

        v = jnp.reshape(v, (B, T, self.n_head, C // self.n_head))  # B, T, n_head, C // n_head

        y = self._apply_attn(q, k, v, mask=mask)

        y = jnp.reshape(y, (B, T, C))  # (B, T, C)
        y = self.c_proj(y)
        return y


class GQ_Attention(SelfAttentionBase):
    def __init__(
        self, config: Config, rngs: nnx.Rngs
    ):
        self.config = config

        self.wq = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=config.init_stddev),
                getattr(config, "attn_wq_kernel_sharding", (None, "model")),
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                getattr(config, "attn_wq_bias_sharding", ("model",)),
            ),
            use_bias=config.attention_bias,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.wkv = nnx.Linear(
            config.n_embed,
            2 * config.n_kv_head * config.n_embed // config.n_head,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=config.init_stddev),
                getattr(config, "attn_wkv_kernel_sharding", (None, "model")),
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                getattr(config, "attn_wkv_bias_sharding", ("model",)),
            ),
            use_bias=config.attention_bias,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.wproj = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(
                    stddev=config.init_stddev * (2 * config.n_layer) ** -0.5
                ),
                getattr(config, "attn_wproj_kernel_sharding", (None, "model")),
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                getattr(config, "attn_wproj_bias_sharding", ("model",)),
            ),
            use_bias=config.attention_bias,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.implementation = config.sdpa_implementation
        self.key_cache = None
        self.value_cache = None


    def _apply_attn(self, q, k, v, mask):
        B, _, C = q.shape
        q = q.reshape((B, -1, self.n_head, C // self.n_head))
        k = k.reshape((B, -1, self.n_kv_head, C // self.n_head))
        v = v.reshape((B, -1, self.n_kv_head, C // self.n_head))
        y = _calc_attention(
            q, k, v, implementation=self.implementation, mask=mask
        )  # (B, T, n_head, C // n_head)
        y = jnp.reshape(y, (B, -1, C))
        return y


    def __call__(self, x, mask=None):
        B, x_T, C = x.shape
        q = self.wq(x)  # (B, x_T, C)
        kv = self.wkv(x)  # (B, x_T, 2 * n_kv_head * C // n_head)
        k, v = jnp.split(
            kv, 2, axis=-1
        )  # (B, x_T, n_kv_head * C // n_head)

        if self.config.use_cache is True and self.key_cache is not None:
            q_prev = jnp.zeros((B, self.key_cache.shape[1], C), dtype=jnp.bfloat16)
            q = jnp.concat([q_prev, q], axis=1)
            k = jnp.concat((self.key_cache, k), axis=1) 
            v = jnp.concat((self.value_cache, v), axis=1)

        y = self._apply_attn(
            q, k, v, mask=mask
        )  # (B, T, n_head, C // n_head)

        if self.config.use_cache is True and self.key_cache is not None: 
            y = y[:, -1:, ...]

        y = self.wproj(y)

        self.key_cache = k[:, -self.config.block_size:, :] # truncate if bigger than block size
        self.value_cache = v[:, -self.config.block_size:, :]

        return y


class GQ_Attention_w_RoPE(GQ_Attention, RoPE_Llama):
    def __init__(self, config: Config, rope_omega: nnx.Variable, rngs: nnx.Rngs):
        GQ_Attention.__init__(self, config, rngs) 
        RoPE_Llama.__init__(self, omega=rope_omega)


    def _apply_attn(self, q, k, v, mask):
        B, _, C = q.shape
        q = q.reshape((B, -1, self.n_head, C // self.n_head))
        k = k.reshape((B, -1, self.n_kv_head, C // self.n_head))
        v = v.reshape((B, -1, self.n_kv_head, C // self.n_head))
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        y = _calc_attention(
            q, k, v, implementation=self.implementation, mask=mask
        )  # (B, T, n_head, C // n_head)
        y = jnp.reshape(y, (B, -1, C))
        return y


class CausalSelfAttention_w_RoPE(CausalSelfAttention, RoPE_Llama):
    def __init__(self, config: Config, rope_omega: nnx.Variable, rngs: nnx.Rngs):
        CausalSelfAttention.__init__(self, config, rngs)
        RoPE_Llama.__init__(self, omega=rope_omega)

    def _apply_attn(self, q, k, v, mask=None):
        # Apply RoPE to q and k
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        y = _calc_attention(
            q, k, v, implementation=self.implementation, mask=mask
        )
        return y

