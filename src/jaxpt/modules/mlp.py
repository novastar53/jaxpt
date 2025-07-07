from functools import partial

import jax
import flax.nnx as nnx
import jax.numpy as jnp

from jax.sharding import PartitionSpec

from jaxpt.modules.config import Config


class Experts(nnx.Module):
    def __init__(self, config, rngs):
        w_c_fc_sharding = getattr(config, "experts_w_c_fc_sharding", ("device",))
        b_c_fc_sharding = getattr(config, "experts_b_c_fc_sharding", ("device",))
        w_gate_sharding = getattr(config, "experts_w_gate_sharding", ("device",))
        b_gate_sharding = getattr(config, "experts_b_gate_sharding", ("device",))
        w_c_proj_sharding = getattr(config, "experts_w_c_proj_sharding", ("device",))
        b_c_proj_sharding = getattr(config, "experts_b_c_proj_sharding", ("device",))


        w_c_fc_init = nnx.with_partitioning(
            nnx.initializers.normal(stddev=0.02), sharding=w_c_fc_sharding
        )

        b_init = nnx.with_partitioning(
            nnx.initializers.zeros, sharding=b_c_fc_sharding
        )

        w_c_proj_init = nnx.with_partitioning(
            nnx.initializers.normal(stddev=0.02 * (2 * config.n_layer) ** -0.5),
            sharding=w_c_proj_sharding,
        )

        self.w_c_fc = nnx.Param(
            w_c_fc_init(
                rngs.default(),
                (config.n_experts, config.n_embed, config.n_mlp_hidden),
            )
        )
        self.b_c_fc = nnx.Param(
            b_init(rngs.default(), (config.n_experts, 1, config.n_mlp_hidden))
        )

        self.w_gate = nnx.Param(
            nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02), sharding=w_gate_sharding
            )(
                rngs.default(),
                (config.n_experts, config.n_embed, config.n_mlp_hidden),
            )
        )
        self.b_gate = nnx.Param(
            nnx.with_partitioning(
                nnx.initializers.zeros, sharding=b_gate_sharding
            )(rngs.default(), (config.n_experts, 1, config.n_mlp_hidden))
        )

        self.w_c_proj = nnx.Param(
            w_c_proj_init(
                rngs.default(),
                (config.n_experts, config.n_mlp_hidden, config.n_embed),
            )
        )
        self.b_c_proj = nnx.Param(
            nnx.with_partitioning(
                nnx.initializers.zeros, sharding=b_c_proj_sharding
            )(rngs.default(), (config.n_experts, 1, config.n_embed))
        )

    def __call__(self, x):
        spec = PartitionSpec("device", )
        x = jax.lax.with_sharding_constraint(x, spec)
        h = jnp.einsum("eti,eih->eth", x, self.w_c_fc) + self.b_c_fc
        g = jnp.einsum("eti,eih->eth", x, self.w_gate) + self.b_gate
        g = nnx.silu(g)
        og = jnp.einsum("eth,eth->eth", h, g)
        o = jnp.einsum("eth,eho->eto", og, self.w_c_proj) + self.b_c_proj
        o = jax.lax.with_sharding_constraint(o, spec)
        return o


class MOE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.router_gate = nnx.Linear(
            config.n_embed,
            config.n_experts,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02), sharding=(None,)
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros, sharding=(None,)
            ),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.experts = Experts(config, rngs)
        self.top_k = config.top_k
        self.n_experts = config.n_experts
        self.load_factor = config.load_factor
        self.add_noise = False
        self.rngs = rngs

    def __call__(self, x):
        spec = PartitionSpec("device", )
        B, T, C = x.shape
        x = x.reshape(-1, C)
        logits = self.router_gate(x)  # B, n_experts
        logits = jax.lax.with_sharding_constraint(logits, spec)  # B, n_experts
        # if self.add_noise:
        #    logits += 0.01 * jax.random.normal(key=self.rngs.gate_noise(), shape=logits.shape)

        top_k_logits, expert_indices = jax.lax.top_k(
            logits, self.top_k
        )  # B, top_k
        zeros = jnp.full_like(logits, float("-inf"))  # B, n_experts
        zeros = jax.lax.with_sharding_constraint(zeros, spec)
        sparse_logits = jnp.put_along_axis(
            zeros, expert_indices, top_k_logits, axis=-1, inplace=False
        )  # B, n_experts
        sparse_logits = jax.lax.with_sharding_constraint(sparse_logits, spec)
        expert_probs = jax.nn.softmax(sparse_logits, axis=-1)  # B, n_experts
        expert_probs = jax.lax.with_sharding_constraint(expert_probs, spec)

        expert_indices_mask = jax.nn.one_hot(
            expert_indices, num_classes=self.n_experts, axis=-1
        )  # B, n_experts, 2
        expert_indices_mask = jax.lax.with_sharding_constraint(
            expert_indices_mask, spec
        )
        expert_indices_mask = jnp.sum(
            expert_indices_mask, axis=1
        )  # B, n_experts
        expert_token_positions = (
            jnp.cumsum(expert_indices_mask, axis=0) * expert_indices_mask
        )  # B, n_experts

        expert_token_positions = jax.lax.with_sharding_constraint(
            expert_token_positions, spec
        )
        expert_input_experts, expert_input_token_idxs = jnp.nonzero(
            expert_token_positions.T, size=B * self.top_k
        )  # B * top_k, B * top_k
        expert_input_positions = (
            jnp.int32(
                expert_token_positions.T[
                    expert_input_experts, expert_input_token_idxs
                ]
            )
            - 1
        )  # B * top_k
        expert_input_positions = jax.lax.with_sharding_constraint(
            expert_input_positions, spec
        )

        expert_inputs = jnp.zeros((self.n_experts, self.top_k * B, C))
        expert_inputs = jax.lax.with_sharding_constraint(expert_inputs, spec)
        expert_inputs = expert_inputs.at[
            expert_input_experts, expert_input_positions
        ].set(x[expert_input_token_idxs])
        input_counters = jnp.max(expert_input_positions, axis=0)

        f = input_counters / B
        P = jnp.mean(expert_probs, axis=0)
        aux_loss = jnp.sum(f * P) / (self.n_experts**2)

        expert_outputs = self.experts(
            expert_inputs
        )  # n_experts, expert_capacity
        expert_outputs = jax.lax.with_sharding_constraint(expert_outputs, spec)

        y = jnp.zeros_like(x)
        y = jax.lax.with_sharding_constraint(y, spec)
        y = y.at[expert_input_token_idxs].add(
            expert_outputs[expert_input_experts, expert_input_positions]
            * expert_probs[expert_input_token_idxs, expert_input_experts][
                ..., None
            ]
        )
        y = jax.lax.with_sharding_constraint(y, spec)

        return y, 0


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
                mesh=config.mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros, sharding=fc_bias_sharding, mesh=config.mesh
            ),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.gate = nnx.Linear(
            config.n_embed,
            config.n_mlp_hidden,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                sharding=gate_kernel_sharding,
                mesh=config.mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros, sharding=gate_bias_sharding, mesh=config.mesh
            ),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
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
                mesh=config.mesh,
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros, sharding=proj_bias_sharding, mesh=config.mesh
            ),
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
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.c_fc(x)
        x = nnx.gelu(x, approximate=True)
        x = self.c_proj(x)
        return x
