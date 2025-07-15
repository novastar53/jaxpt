from functools import partial

import flax.nnx as nnx

from jaxpt.modules.config import Config


class GLU(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        fc_kernel_sharding = getattr(config, "glu_fc_kernel_sharding", (None, "model"))
        fc_bias_sharding = getattr(config, "glu_fc_bias_sharding", ("model",))
        gate_kernel_sharding = getattr(config, "glu_gate_kernel_sharding", (None, "model"))
        gate_bias_sharding = getattr(config, "glu_gate_bias_sharding", ("model",))
        proj_kernel_sharding = getattr(config, "glu_proj_kernel_sharding", ("model", None))
        proj_bias_sharding = getattr(config, "glu_proj_bias_sharding", (None,))

        self.c_fc = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                sharding=fc_kernel_sharding,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros, sharding=fc_bias_sharding, 
            ),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.gate = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                sharding=gate_kernel_sharding,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros, sharding=gate_bias_sharding, 
            ),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_mlp_hidden,
            config.n_embed,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(
                    stddev=0.02 * (2 * config.n_layer) ** -0.5
                ),
                sharding=proj_kernel_sharding,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros, sharding=proj_bias_sharding, 
            ),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
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
        fc_kernel_sharding = getattr(config, "mlp_fc_kernel_sharding", (None, "model"))
        fc_bias_sharding = getattr(config, "mlp_fc_bias_sharding", ("model",))
        proj_kernel_sharding = getattr(config, "mlp_proj_kernel_sharding", ("model", None))
        proj_bias_sharding = getattr(config, "mlp_proj_bias_sharding", (None,))

        self.c_fc = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02), sharding=fc_kernel_sharding
            ),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, fc_bias_sharding),
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_mlp_hidden,
            config.n_embed,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(
                    stddev=0.02 * (2 * config.n_layer) ** -0.5
                ),
                proj_kernel_sharding,
            ),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, proj_bias_sharding),
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.c_fc(x)
        x = nnx.gelu(x, approximate=True)
        x = self.c_proj(x)
        return x
