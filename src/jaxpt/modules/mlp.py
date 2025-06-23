from functools import partial

import jax
import flax.nnx as nnx
import jax.numpy as jnp

from jaxpt.modules.config import Config


class MOE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.experts = [ GLU(config, rngs) for _ in range(config.n_experts) ]
        self.router_gate = nnx.Linear(
            config.n_embed, 
            config.n_experts, 
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(
                stddev=0.02), (None, "model")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("model",)),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.config = config


    def __call__(self, x):
        expert_weights, expert_indices = self.router_gate(x) # obtain the expert indices and weights for each token
        final_output = jnp.zeros_like(x) # create a zero array for the final combined output from the top_k experts

        # Reshape inputs for batch processing
        flat_x = x.reshape(-1, x.shape[-1]) # flatten the batch and sequence dimensions (why?) 
        flat_expert_weights = expert_weights.reshape(-1, expert_weights.shape[-1]) # flatten the expert weights

        # This for loop will be unrolled during lowering.
        # Since the number of experts is fixed, it's a constant time operation.
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (expert_indices == i).any(axis=-1)
            flat_mask = expert_mask.reshape(-1)

            idxs = jnp.nonzero(flat_mask, size=flat_mask.size)[0]   # shape (n_trues,)

            expert_input = flat_x[flat_mask]
            expert_output = expert(expert_input)

            # Extract and apply gating scores
            gating_scores = flat_expert_weights[flat_mask, i]
            gating_scores = gating_scores[..., None]
            weighted_output = expert_output * gating_scores

            # Update final output additively by indexing and adding
            final_output = final_output.at[expert_mask].set(weighted_output)

        return final_output


class GLU(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                (None, "model")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("model,")),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.gate = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                (None, "model")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("model",)),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_mlp_hidden,
            config.n_embed,
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ), ("model", None)),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros, (None,)),
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
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                (None, "model")),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                ("model",)),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_mlp_hidden,
            config.n_embed,
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ), ("model", None)),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros, (None,)),
            dtype=config.dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.c_fc(x)
        x = nnx.gelu(x, approximate=True)
        x = self.c_proj(x)
        return x
