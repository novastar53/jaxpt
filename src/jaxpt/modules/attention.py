from typing import Literal
import abc
import math

import jax
from jax._src.cudnn.fused_attention_stablehlo import (
    dot_product_attention as cudnn_dot_product_attention, MaskType)
import flax.nnx as nnx
import jax.numpy as jnp

from jaxpt.modules import Config


def _calc_slow_attn(q, k, v, mask, bias = None):
    B, T, n_head, hs = q.shape

    if bias == None: 
        bias = jnp.zeros(shape=(T, T), dtype=q.dtype)

    q = jnp.transpose(q, axes=(0, 2, 1, 3)) # (B, n_head, T, hs)
    k = jnp.transpose(k, axes=(0, 2, 3, 1)) # (B, n_head, hs, T)
    v = jnp.transpose(v, axes=(0, 2, 1, 3)) # (B, n_head, T, hs)
    att = ( q @ k ) + bias
    att = att / math.sqrt(k.shape[-1])  # B, n_head, T, T
    att = jnp.where(mask[:, :, :T, :T] == 0.0, float('-inf'), att)
    att = jax.nn.softmax(att, axis=-1)
    y = att @ v # (B, n_head, T, T) x (b, n_head, T, hs) -> (B, n_head, T, hs)
    y = jnp.transpose(y, axes=(0, 2, 1, 3)) # (B, T, n_head, hs)
    return y, att

def _calc_attention(
    query,
    key,
    value,
    mask = None,
    bias = None,
    implementation: Literal['xla', 'cudnn'] | None = None):

    output_shape = jnp.asarray(query).shape
    _, _, _, H = key.shape
    scale_val = (1.0 / jnp.sqrt(H)) 

    match implementation:
        case 'xla':
            out = jax.nn._dot_product_attention_xla(
                query, key, value, None, None, is_causal=True,
                scale=scale_val, q_seqlen=None,
                kv_seqlen=None,
                local_window_size=None,
            )
        case 'cudnn':
            out = cudnn_dot_product_attention(
                query, key, value, bias, None, None,
                None, scale=scale_val, mask_type=MaskType.CAUSAL,
                sliding_window_length=None,
            )
        case _: 
           out, _ = _calc_slow_attn(query, key, value, mask, bias)

    return jnp.reshape(out, output_shape)

class SelfAttentionBase(nnx.Module, abc.ABC):
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
        self.mask = nnx.Variable(
            jnp.tril(
                jnp.ones(
                    (config.block_size, config.block_size), dtype=config.dtype
                ).reshape((1, 1, config.block_size, config.block_size))
            )
        )
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.implementation = config.sdpa_implementation
        self.use_slow_attention = config.use_slow_attention

    @abc.abstractmethod
    def __call__(self, x):
        raise NotImplementedError

class CausalSelfAttention(SelfAttentionBase):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)

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

        y = _calc_attention(
            q, k, v, implementation=self.implementation
        )

        y = jnp.reshape(y, (B, T, C))  # (B, T, C)
        y = self.c_proj(y)
        return y

class RoPEAttention(SelfAttentionBase):
    def __init__(self, config: Config, omega: nnx.Variable, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.omega = omega

   
    def apply_rope(self, v):
        omega = self.omega[:v.shape[-2], :]
        a = v * jnp.cos(omega) 
        b = v * jnp.sin(omega) 
        b = b.reshape(-1, 2)[..., ::-1]
        b = b.at[..., -1].multiply(-1).reshape(v.shape)
        return a + b

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)  # B, T, 3 * C
        q, k, v = jnp.split(qkv, 3, axis=-1)  # 3 * (B, T, C)

        q = jnp.reshape(
            q, (B, self.n_head, T, C // self.n_head)
        )  
        q = self.apply_rope(q)
        q = jnp.reshape(
            q, (B, T, self.n_head, C // self.n_head)
        )

        k = jnp.reshape(
            k, (B, self.n_head, T, C // self.n_head)
        )  
        k = self.apply_rope(k)
        k = jnp.reshape(
            k, (B, T, self.n_head, C // self.n_head)
        )

        v = jnp.reshape(
            v, (B, T, self.n_head, C // self.n_head)
        )  # B, T, n_head, C // n_head

        y = _calc_attention(
            q, k, v, implementation=self.implementation
        )

        y = jnp.reshape(y, (B, T, C))  # (B, T, C)
        y = self.c_proj(y)
        return y


