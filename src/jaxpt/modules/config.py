from dataclasses import dataclass

import jax.numpy as jnp

@dataclass
class Config:
    name: str = "gpt"
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
