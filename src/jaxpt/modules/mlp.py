import flax.nnx as nnx

from jaxpt.modules import Config

class GLUMLP(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embed,
            4 * config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.gate = nnx.Linear(
            config.n_embed,
            4 * config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            4 * config.n_embed,
            config.n_embed,
            kernel_init=nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.c_fc(x)
        g = nnx.sigmoid(self.gate(x))
        x = x * g
        x = self.c_proj(x)
        return x



class MLP(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embed,
            4 * config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            4 * config.n_embed,
            config.n_embed,
            kernel_init=nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.c_fc(x)
        x = nnx.gelu(x, approximate=True)
        x = self.c_proj(x)
        return x

