from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from jaxpt.modules.attention import CausalSelfAttention, GQ_Attention
from jaxpt.modules.config import Config

@dataclass
class Attn_Config(Config):
    n_layer: int = 1
    block_size: int = 5  # sequence length
    n_head: int = 4  # number of attention heads
    n_kv_head: int = 2  # number of key-value heads
    n_embed: int = 12  # number token embedding dimensionsa
    sdpa_implementation: str = "slow"
    init_stddev: float = 0.02
    attention_bias: bool = False  # use bias in attention layers
    use_cache: bool = False
 

def test_causal_self_attention():
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="xla"
    )
    x = jax.random.normal(jax.random.key(0), (batch_size, config.block_size, config.n_embed))
    attn = CausalSelfAttention(
        config=config,
        rngs=nnx.Rngs(default=0))
    y1 = attn(x)

    config = Attn_Config(
        sdpa_implementation="slow"
    )
    attn = CausalSelfAttention(
        config=config,
        rngs=nnx.Rngs(default=0))
    y2 = attn(x)

    assert(jnp.allclose(y1, y2, atol=1e-5))


def test_gq_attention():
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="xla"
    )
    x = jax.random.normal(jax.random.key(0), (batch_size, config.block_size, config.n_embed))
    attn = GQ_Attention(
        config=config,
        rngs=nnx.Rngs(default=0))
    y1 = attn(x)
    config = Attn_Config(
        sdpa_implementation="slow"
    )
    attn = GQ_Attention(
        config=config,
        rngs=nnx.Rngs(default=0))
    y2 = attn(x)

    assert(jnp.allclose(y1, y2, atol=1e-5))



if __name__ == "__main__":

    #test_causal_self_attention()
    test_gq_attention()