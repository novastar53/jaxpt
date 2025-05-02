import jax.numpy as jnp


class RoPE:
    def __init__(self, omega):
        self.omega = omega

    def apply_rope(self, v):
            omega = self.omega[:v.shape[-2], :]
            a = v * jnp.cos(omega) 
            b = v * jnp.sin(omega) 
            b = b.reshape(-1, 2)[..., ::-1]
            b = b.at[..., -1].multiply(-1).reshape(v.shape)
            return a + b