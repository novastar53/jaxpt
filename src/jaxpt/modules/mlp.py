from functools import partial

import flax.nnx as nnx

from jaxpt.modules.config import Config

class MOE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden * config.n_experts,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.gate = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden * config.n_experts,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_mlp_hidden * config.n_experts,
            config.n_embed * config.n_experts,
            kernel_init=nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ),
            bias_init=nnx.initializers.zeros,
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        match config.glu_activation:
            case "sigmoid":
                self.activation = nnx.sigmoid
            case "gelu":
                self.activation = partial(nnx.gelu, approximate=True)
            case "swish" | "silu":
                self.activation = nnx.silu
            case _:
                self.activation = nnx.sigmoid

        self.moe_gate = nnx.Linear(
            config.n_embed, 
            config.n_experts, 
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.config = config
    
    def __call__(self, x):
        gate = self.moe_gate(x) # B x T x nE
        gate = nnx.softmax(gate, axis=-1) # B x T x nE
        _, indices  = jax.lax.top_k(gate, config.top_k)

        in_dims, out_dims  = self.c_fc.kernel.shape
        c_fc = self.c_fc.kernel.reshape(in_dims, self.config.n_experts, out_dims)
        c_fc = jnp.take(c_fc, indices, axis=1)

        in_dims, out_dims = 

        h = self.c_fc(x)
        g = self.gate(x)
        g = self.activation(g)
        h = g * h
        y = self.c_proj(h)
 
        return x


class GLU(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.gate = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_mlp_hidden,
            config.n_embed,
            kernel_init=nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ),
            bias_init=nnx.initializers.zeros,
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        match config.glu_activation:
            case "sigmoid":
                self.activation = nnx.sigmoid
            case "gelu":
                self.activation = partial(nnx.gelu, approximate=True)
            case "swish" | "silu":
                self.activation = nnx.silu
            case _:
                self.activation = nnx.sigmoid

    def __call__(self, x):
        h = self.c_fc(x)
        g = self.gate(x)
        g = self.activation(g)
        h = g * h
        y = self.c_proj(h)
        return y


class MLP(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_mlp_hidden,
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
