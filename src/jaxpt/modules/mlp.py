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
        self.c_proj = nnx.Linear(
            config.n_mlp_hidden * config.n_experts,
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
        B, T, C = x.shape
        H = self.config.n_mlp_hidden
        nE = self.config.n_experts
        K = self.config.n_top_k_experts

        gate = self.moe_gate(x) # B x T x nE
        gate_probs = nnx.softmax(gate, axis=-1) # B x T x nE
        gate_top_k_probs, gate_top_k_indices  = jax.lax.top_k(gate, K)

        c_fc_kernel = self.c_fc.kernel.reshape(C, H, nE) # (C x H x nE)
        c_fc_top_k = c_fc_kernel[:, :, gate_top_k_indices].reshape(B, T, C, H, nE)
        h = jnp.einsum("btc,btchk->bthk", x, c_fc_top_k) # (B, T, H, K)

        h = self.activation(h)

        c_proj_kernel = self.c_proj.kernel.reshape(H, C, nE)
        c_proj_top_k = c_proj_kernel[:, :, idxs].reshape(B, T, H, C, K)
        o1 = jnp.einsum("bthp,bthck->btck", h, c_proj_top_k)
        o = jnp.einsum("btck,btk->btc", o1, gate_top_k_probs)

        return o


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
