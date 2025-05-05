import flax.nnx as nnx
import jax.numpy as jnp


def calc_rope_omega(n_embed: int, n_head: int, block_size: int,
                    rope_base_freq: float, dtype: jnp.dtype) -> nnx.Variable:
    query_size = n_embed // n_head
    pow = jnp.arange(0, query_size, 2)
    omega = rope_base_freq**(pow/query_size)
    omega = jnp.expand_dims(omega, axis=0)
    omega = jnp.repeat(omega, 2, axis=1)
    omega = jnp.repeat(omega, block_size, axis=0)
    pos = jnp.arange(0, block_size)
    pos = jnp.expand_dims(pos, axis=1)
    omega = omega * pos
    return nnx.Variable(omega)



class RoPE:
    def __init__(self, omega: nnx.Variable):
        self.omega = omega

    def apply_rope(self, v):
            v = v.swapaxes(1, 2)
            omega = self.omega[:v.shape[-2], :]
            a = v * jnp.cos(omega) 
            b = v * jnp.sin(omega) 
            b = b.reshape(-1, 2)[..., ::-1]
            b = b.at[..., -1].multiply(-1).reshape(v.shape)
            y = a + b
            y = y.swapaxes(1, 2)
            return y