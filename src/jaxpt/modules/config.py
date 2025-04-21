from abc import ABC
from dataclasses import dataclass

import jax.numpy as jnp

@dataclass
class Config(ABC):
    dtype: jnp.dtype = jnp.float32
    block_size: int = 1024  # sequence length
    vocab_size: int = 50304  # 50257 padded to the nearest multiple of 64
    n_layer: int = 12  # number of attention blocks
    n_head: int = 12  # number of attention heads
    n_embed: int = 768  # number token embedding dimensionsa
