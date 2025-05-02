from typing import Literal
import abc
import math

import jax
import flax.nnx as nnx
import jax.numpy as jnp

from jaxpt.modules.config import Config
from jaxpt.modules.position import RoPE


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
    implementation: Literal['xla', 'cudnn', 'slow'] | None = None):

    output_shape = jnp.asarray(query).shape
    _, _, _, H = key.shape
    scale_val = (1.0 / jnp.sqrt(H)) 

    match implementation:
        case 'xla' | 'cudnn':
            out = jax.nn.dot_product_attention(
                query, key, value, mask=mask, bias=bias, 
                is_causal=True, implementation=implementation
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
        if config.sdpa_implementation == "slow":
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


class RoPEAttention(SelfAttentionBase, RoPE):
    def __init__(self, config: Config, rope_omega: nnx.Variable, rngs: nnx.Rngs):
        SelfAttentionBase.__init__(self, config=config, rngs=rngs)
        RoPE.__init__(self, omega=rope_omega)

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


class MQ_Attention(SelfAttentionBase, RoPE):
    def __init__(self, config: Config, rope_omega: nnx.Variable, rngs: nnx.Rngs):
        #SelfAttentionBase.__init__(self, config=config, rngs=rngs)
        RoPE.__init__(self, omega=rope_omega)

        self.wq = nnx.Linear(
            config.n_embed,
            config.n_embed  // config.n_head,
            kernel_init=nnx.initializers.normal(stddev=config.init_stddev),
            dtype=config.dtype,
            rngs=rngs
        )
        self.wkv = nnx.Linear(
            config.n_embed,
            2 * config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=config.init_stddev),
            bias_init=nnx.initializers.zeros,
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
            dtype=config.dtype,
            rngs=rngs,
        )
        self.n_head = config.n_head
        self.implementation = config.sdpa_implementation
    
    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x) # (B, T, C // n_head)
        kv = self.wkv(x) # (B, T, 2 * C)
        k, v = jnp.split(kv, 2, axis=-1) # (B, T, 2 * C)

        q = jnp.reshape(
            q, (B, 1, T, C // self.n_head)
        )  
        q = self.apply_rope(q)
        q = jnp.broadcast_to(
            q.squeeze(), (self.n_head, B, T, C // self.n_head)
        )
        q = jnp.reshape(q, (B, T, self.n_head, C // self.n_head))


        k = jnp.reshape(
            k, (B, self.n_head, T, C // self.n_head)
        )  
        k = self.apply_rope(k)
        k = jnp.reshape(
            k, (B, T, self.n_head, C // self.n_head)
        )

        v = jnp.reshape(v, (B, T, self.n_head, C // self.n_head))

        y = _calc_attention(
            q, k, v, implementation=self.implementation
        ) # (B, T, n_head, C // n_head)

        y = jnp.reshape(y, (B, T, C))
        y = self.wproj(y) 
        return y


class GQ_Attention(SelfAttentionBase, RoPE):
    def __init__(self, config: Config, rope_omega: nnx.Variable, rngs: nnx.Rngs):
        #SelfAttentionBase.__init__(self, config=config, rngs=rngs)
        RoPE.__init__(self, omega=rope_omega)

        self.wq = nnx.Linear(
            config.n_embed,
            config.n_config.n_embed  // config.n_head,
            kernel_init=nnx.initializers.normal(stddev=config.init_stddev),
            dtype=config.dtype,
            rngs=rngs
        )
        self.wkv = nnx.Linear(
            config.n_embed,
            2 * config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=config.init_stddev),
            bias_init=nnx.initializers.zeros,
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
            dtype=config.dtype,
            rngs=rngs,
        )
        self.n_head = config.n_head
        self.implementation = config.sdpa_implementation
    
    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x) # (B, T, C // n_head)
        kv = self.wkv(x) # (B, T, 2 * C)
        k, v = jnp.split(kv, 2, axis=-1) # (B, T, 2 * C)

        q = jnp.reshape(
            q, (B, 1, T, C // self.n_head)
        )  
        q = self.apply_rope(q)
        q = jnp.broadcast_to(
            q, (self.n_head, B, T, C // self.n_head)
        )
        q = jnp.reshape(q, (B, T, self.n_head, C // self.n_head))

        k = jnp.reshape(
            k, (B, self.n_head, T, C // self.n_head)
        )  
        k = self.apply_rope(k)
        k = jnp.reshape(
            k, (B, T, self.n_head, C // self.n_head)
        )

        v = jnp.reshape(v, (B, T, self.n_head, C // self.n_head))

        y = _calc_attention(
            q, k, v, implementation=self.implementation
        ) # (B, T, n_head, C // n_head)

        y = jnp.reshape(y, (B, T, C))
        y = self.wproj(y) 
        return y