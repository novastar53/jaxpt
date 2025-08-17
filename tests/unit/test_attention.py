from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from jaxpt.modules.attention import CausalSelfAttention, GQ_Attention
from jaxpt.modules.config import Config

@dataclass
class Attn_Config(Config):
    n_layer: int = 1
    block_size: int = 2048  # sequence length
    n_head: int = 12  # number of attention heads
    n_kv_head: int = 4  # number of key-value heads
    n_embed: int = 576  # number token embedding dimensionsa
    sdpa_implementation: str = "slow"
    init_stddev: float = 0.02
    attention_bias: bool = False  # use bias in attention layers
    use_cache: bool = False
 

def test_causal_self_attention():
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="xla"
    )
    attn = CausalSelfAttention(
        config=config,
        rngs=nnx.Rngs(default=0))
    y1 = attn(
        jnp.ones((batch_size, config.block_size, config.n_embed))
    )
    config = Attn_Config(
        sdpa_implementation="slow"
    )
    attn = CausalSelfAttention(
        config=config,
        rngs=nnx.Rngs(default=0))
    y2 = attn(
        jnp.ones((batch_size, config.block_size, config.n_embed))
    )
    assert(jnp.allclose(y1, y2))


def test_gq_attention():
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="xla"
    )
    attn = GQ_Attention(
        config=config,
        rngs=nnx.Rngs(default=0))
    y1 = attn(
        jnp.ones((batch_size, config.block_size, config.n_embed))
    )
    config = Attn_Config(
        sdpa_implementation="slow"
    )
    attn = GQ_Attention(
        config=config,
        rngs=nnx.Rngs(default=0))
    y2 = attn(
        jnp.ones((batch_size, config.block_size, config.n_embed))
    )

    assert(jnp.allclose(y1, y2))



if __name__ == "__main__":

    test_causal_self_attention()
    test_gq_attention()