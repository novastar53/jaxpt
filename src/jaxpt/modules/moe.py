import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax.nnx.nn import dtypes
from jax.sharding import PartitionSpec

from jaxpt.modules.config import Config


spec = PartitionSpec("devices",)


class Experts(nnx.Module):
    def __init__(self, config, rngs):
        w_c_fc_init = nnx.with_partitioning(
            nnx.initializers.normal(stddev=0.02),
            sharding=spec)
        
        b_init = nnx.with_partitioning(
            nnx.initializers.zeros,
            sharding=spec)
        
        w_c_proj_init = nnx.with_partitioning(
            nnx.initializers.normal(stddev=0.02 * (2 * config.n_layer) ** -0.5),
            sharding=spec
        )

        self.w_c_fc = nnx.Param(w_c_fc_init(rngs.default(),
            (
                config.n_experts,
                config.n_embed,
                config.n_mlp_hidden
            ),
            config.param_dtype
        ))
        self.b_c_fc = nnx.Param(b_init(rngs.default(),
        (
            config.n_experts,
            1,
            config.n_mlp_hidden
        ), config.param_dtype))

        self.w_gate = nnx.Param(w_c_fc_init(rngs.default(),
        (
            config.n_experts,
            config.n_embed,
            config.n_mlp_hidden
        ), config.param_dtype))
        self.b_gate = nnx.Param(b_init(rngs.default(),
        (
            config.n_experts,
            1,
            config.n_mlp_hidden
        ), config.param_dtype))

        self.w_c_proj = nnx.Param(
            w_c_proj_init(
                rngs.default(),
                (
                    config.n_experts,
                    config.n_mlp_hidden,
                    config.n_embed
                ), config.param_dtype)
        )
        self.b_c_proj = nnx.Param(
            b_init(
                rngs.default(),
                (
                    config.n_experts,
                    1,
                    config.n_embed
                ),
                config.param_dtype
            )
        )
        self.config = config

    def __call__(self, x):
        (x, w_c_fc, b_c_fc, w_gate, 
        b_gate, w_c_proj, b_c_proj) = dtypes.promote_dtype(
        (x, self.w_c_fc.value, self.b_c_fc.value, self.w_gate.value, 
        self.b_gate.value, self.w_c_proj.value, self.b_c_proj.value), dtype=self.config.dtype
        )
        x = jax.lax.with_sharding_constraint(x, spec)
        h = jnp.einsum('eti,eih->eth', x, w_c_fc) + b_c_fc
        g = jnp.einsum('eti,eih->eth', x, w_gate) + b_gate
        o = jnp.einsum('eth,eho->eto', nnx.silu(h * g), w_c_proj) + b_c_proj
        o = jax.lax.with_sharding_constraint(o, spec)
        return o


class MOE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.router_gate = nnx.Linear(
            config.n_embed,
            config.n_experts,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                sharding=(None,)),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros,
            sharding=(None,)),
            use_bias=config.mlp_bias,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.experts = Experts(config, rngs)
        self.top_k = config.top_k
        self.n_experts = config.n_experts
        self.load_factor = config.load_factor
        self.add_noise = False
        self.aux_loss = False
        self.rngs = rngs


    def _get_expert_inputs(self, x, logits, expert_capacity):
        T, _ = logits.shape
        _, C = x.shape
        top_k_logits, expert_indices = jax.lax.top_k(logits, self.top_k) # T, top_K
        zeros = jnp.full_like(logits, float('-inf')) # T, n_experts
        expert_probs = jax.nn.softmax(top_k_logits, axis=-1) # T, n_experts  

        # Swap the sequence (T) and top_k dimensions so that when the array is
        # flattened, the higher ranked experts appear first.
        expert_indices = jnp.swapaxes(expert_indices, 0, 1).ravel() # top_K * T
        # Calculate the expert buffer positions all the tokens in the batch
        expert_one_hot = jax.nn.one_hot(expert_indices, self.n_experts, dtype=jnp.int32) # top_K * T, n_experts
        expert_positions = (jnp.cumsum(expert_one_hot, axis=0) * expert_one_hot) - 1 # top_K * T, n_experts
        # Reshape the buffer index to match the original ordering and dimensions
        expert_positions = expert_positions.reshape(-1, T, self.n_experts) # top_K, T, n_experts
        expert_positions = jnp.swapaxes(expert_positions, 0, 1) # T, top_K, n_experts
        # Extract the buffer positions for each token and k, 
        expert_positions = jnp.max(expert_positions, axis=2) # T, top_K
        # Restore the shape and order of expert_indices
        expert_indices = jnp.swapaxes(expert_indices.reshape(-1, T), 0, 1) # T, top_K

        zeros = jnp.zeros((self.n_experts, expert_capacity, C)) # n_experts, expert_cap, C

        x = jnp.repeat(x, self.top_k, axis=0)
        expert_inputs = zeros.at[expert_indices.ravel(), 
                                 expert_positions.ravel()].add(x)

        return expert_probs, expert_positions, expert_indices, expert_inputs


    def _collect_outputs(self, expert_outputs, expert_indices, expert_positions, 
                        top_k_probs):
        T, K, C = expert_outputs.shape
        expert_outputs = expert_outputs[expert_indices, expert_positions]
        expert_outputs = jnp.sum(top_k_probs[..., None] * expert_outputs, axis=1)
        return expert_outputs

    def _add_random_noise(self, x, key):
        return x + jax.random.normal(key=key, shape=x.shape)

    def __call__(self, x):
        B, T, C = x.shape
        logits = self.router_gate(x) # B, T, n_experts
        if self.add_noise:
            keys = jax.random.split(self.rngs.gate_noise(), self.n_experts)
            logits = jax.vmap(self._add_random_noise)(logits, key)

        expert_capacity = int((1.2 * self.top_k * T ) // self.n_experts)
        (top_k_probs, 
         expert_positions, 
         expert_indices, 
         expert_inputs) = jax.vmap(
            lambda x, l: self._get_expert_inputs(x, l, expert_capacity))(x, logits) # B, n_experts, expert_cap, C 
        
        top_k_probs = jax.lax.with_sharding_constraint(top_k_probs, spec)
        expert_positions = jax.lax.with_sharding_constraint(expert_positions, spec)
        expert_indices = jax.lax.with_sharding_constraint(expert_indices, spec)
        expert_inputs = jax.lax.with_sharding_constraint(expert_inputs, spec) # B, n_experts, expert_cap, C

        if B % self.n_experts == 0:
            expert_inputs = expert_inputs.reshape(self.n_experts, -1, self.n_experts, expert_capacity, C) # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_inputs = jnp.swapaxes(expert_inputs, 0, 2) # n_experts, batch_per_expert, n_experts, expert_cap, C
        else:
            expert_inputs = jnp.swapaxes(expert_inputs, 0, 1) # n_experts, B, expert_cap, C

        expert_inputs = expert_inputs.reshape(-1, C) # n_experts * B * expert_cap, C
        expert_inputs = jax.lax.with_sharding_constraint(expert_inputs, spec)
        expert_inputs = expert_inputs.reshape(self.n_experts, B * expert_capacity, C) # n_experts, B * expert_cap, C

        expert_outputs = self.experts(expert_inputs) # n_experts, B * expert_cap, C
        if B % self.n_experts == 0:
            expert_outputs = expert_outputs.reshape(self.n_experts, -1, self.n_experts, expert_capacity, C) # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_outputs = jnp.swapaxes(expert_outputs, 0, 2) # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_outputs = expert_outputs.reshape(B, self.n_experts, expert_capacity, C) # B, n_experts, expert_cap, C
        else:
            expert_outputs = expert_outputs.reshape(self.n_experts, B, expert_capacity, C) # n_experts, B, expert_cap, C
            expert_outputs = jnp.swapaxes(expert_outputs, 0, 1) # B, n_experts, expert_cap, C

        expert_outputs = jax.lax.with_sharding_constraint(expert_outputs, spec)

        y_pred = jax.vmap(
            lambda x, i, p, prob: self._collect_outputs(x, i, p, prob)
            )(expert_outputs, expert_indices, expert_positions, top_k_probs)

        y_pred = jax.lax.with_sharding_constraint(y_pred, spec)

        if self.aux_loss is True:
            frac = jnp.bincount(expert_indices.flatten(), length=self.n_experts) / (B * T)
            ideal = self.top_k / self.n_experts
            aux_loss = self.n_experts * jnp.sum((frac - ideal) ** 2)
            return y_pred, aux_loss
        
        return y_pred

