import flax.nnx as nnx
import jax.numpy as jnp

def calc_rope_omega(n_embed: int, n_head: int, block_size: int,
                    rope_base_freq: int, dtype: jnp.dtype) -> nnx.Variable:
        query_size = n_embed // n_head
        base_freq = rope_base_freq**(2/query_size)
        omega = jnp.ones((1, query_size), dtype=dtype) * base_freq
        pow = jnp.arange(0, query_size)
        omega = jnp.repeat(omega**pow, block_size, axis=0)
        pos = jnp.arange(0, block_size)
        pos = jnp.expand_dims(pos, axis=1)
        omega = omega * pos
        return nnx.Variable(omega * pos)

class RoPE:
    def __init__(self, omega: nnx.Variable):
        self.omega = omega

    def apply_rope(self, v):
            omega = self.omega[:v.shape[-2], :]
            a = v * jnp.cos(omega) 
            b = v * jnp.sin(omega) 
            b = b.reshape(-1, 2)[..., ::-1]
            b = b.at[..., -1].multiply(-1).reshape(v.shape)
            return a + b