import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax.nnx.nn import dtypes
from jax.sharding import PartitionSpec

from jaxpt.modules.config import Config


spec = PartitionSpec("devices",)


class Experts(nnx.Module):
    def __init__(self, config, rngs):
        self.config = config

        w_c_fc_init = nnx.with_partitioning(
            nnx.initializers.normal(stddev=0.02),
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
        
        self.w_gate = nnx.Param(w_c_fc_init(rngs.default(),
        (
            config.n_experts,
            config.n_embed,
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

        if config.moe_bias:

            b_init = nnx.with_partitioning(
                nnx.initializers.zeros,
                sharding=spec)

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

            self.b_gate = nnx.Param(b_init(rngs.default(),
            (
                config.n_experts,
                1,
                config.n_mlp_hidden
            ), config.param_dtype))

            self.b_c_fc = nnx.Param(b_init(rngs.default(),
            (
                config.n_experts,
                1,
                config.n_mlp_hidden
            ), config.param_dtype))


    def __call__(self, x):
        (x, w_c_fc, w_gate, w_c_proj) = dtypes.promote_dtype(
            (x, self.w_c_fc.value, self.w_gate.value, self.w_c_proj.value), dtype=self.config.dtype
        )
        if self.config.moe_bias:
            (b_c_fc, b_gate, b_c_proj) = dtypes.promote_dtype(
            (self.b_c_fc.value, self.b_gate.value, self.b_c_proj.value), dtype=self.config.dtype
        )
        x = jax.lax.with_sharding_constraint(x, spec)
        h = jnp.einsum('eti,eih->eth', x, w_c_fc)
        if self.config.moe_bias:
            h += b_c_fc
        g = jnp.einsum('eti,eih->eth', x, w_gate)
        if self.config.moe_bias:
            g += b_gate
        g = nnx.silu(g)
        h = g * h
        o = jnp.einsum('eth,eho->eto', h, w_c_proj)
        if self.config.moe_bias:
            o += b_c_proj
        o = jax.lax.with_sharding_constraint(o, spec)
        return o


class MOE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
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
        self.expert_weight_priority = config.expert_weight_priority
        self.add_noise = False
        self.load_balance_loss = False
        self.z_loss = False
        #self.gate_noise_rngstream = rngs['gate_noise'].fork()
        self.gate_noise_rngstream = rngs.gate_noise # TODO: Temporary fix for backward compatibility with Jax version 0.5.2


    def _get_expert_inputs(self, x, gate_probs, expert_capacity):
        T, _ = gate_probs.shape
        _, C = x.shape
        top_k_probs, expert_indices = jax.lax.top_k(gate_probs, self.top_k) # T, top_K

        if self.expert_weight_priority:
            # prioritize expert assignment by expert probs rather than than sequence order
            batch_order = jnp.argsort(top_k_probs[:, 0], descending=True)
            expert_indices = expert_indices[batch_order]

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

        zeros = jnp.zeros((self.n_experts, expert_capacity, C), dtype=x.dtype) # n_experts, expert_cap, C

        x = jnp.repeat(x, self.top_k, axis=0)
        expert_inputs = zeros.at[expert_indices.ravel(), 
                                 expert_positions.ravel()].set(x) # TDOO: This will overwrite tokens if expert capacity is exceeded. 

        if self.expert_weight_priority:
            original_order = jnp.argsort(batch_order)
            expert_indices = expert_indices[original_order]        
            expert_positions = expert_positions[original_order]

        return top_k_probs, expert_positions, expert_indices, expert_inputs


    def _collect_outputs(self, expert_outputs, expert_indices, expert_positions, 
                        top_k_probs):
        expert_outputs = expert_outputs[expert_indices, expert_positions]
        expert_outputs = jnp.sum(top_k_probs[..., None] * expert_outputs, axis=1)
        return expert_outputs


    def __call__(self, x):
        '''
        Expert routing based on the vmoe implementation: 
        https://github.com/google-research/vmoe/blob/main/vmoe/nn/routing.py
        '''
        output = {}
        B, T, C = x.shape
        gate_logits = self.router_gate(x) # B, T, n_experts
        if self.z_loss:
            z_loss = jnp.sum(jnp.square(jnp.log(jnp.sum(jnp.exp(gate_logits), axis=-1)))) / (B * T) 
            output["z_loss"] = z_loss
        if self.add_noise:
            #noise = jax.random.normal(self.gate_noise_rngstream(), 
            #                          gate_logits.shape, dtype=self.config.dtype) * (1/self.n_experts)
            noise = jax.random.uniform(self.gate_noise_rngstream(), gate_logits.shape, 
                                       self.config.dtype, 1-1e-2, 1+1e-2) 
            gate_logits *= noise
        gate_probs = jax.nn.softmax(gate_logits)

        expert_capacity_per_batch = int(self.load_factor * self.top_k * max(1, T / self.n_experts))
        (top_k_probs, 
         expert_positions, 
         expert_indices, 
         expert_inputs) = jax.vmap(
            lambda x, l: self._get_expert_inputs(x, l, expert_capacity_per_batch))(x, gate_probs) # B, n_experts, expert_cap, C 
        
        top_k_probs = jax.lax.with_sharding_constraint(top_k_probs, spec)
        expert_positions = jax.lax.with_sharding_constraint(expert_positions, spec)
        expert_indices = jax.lax.with_sharding_constraint(expert_indices, spec)
        expert_inputs = jax.lax.with_sharding_constraint(expert_inputs, spec) # B, n_experts, expert_cap, C

        if B % self.n_experts == 0:
            expert_inputs = expert_inputs.reshape(self.n_experts, -1, self.n_experts, expert_capacity_per_batch, C) # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_inputs = jnp.swapaxes(expert_inputs, 0, 2) # n_experts, batch_per_expert, n_experts, expert_cap, C
        else:
            expert_inputs = jnp.swapaxes(expert_inputs, 0, 1) # n_experts, B, expert_cap, C

        expert_inputs = expert_inputs.reshape(-1, C) # n_experts * B * expert_cap, C
        expert_inputs = jax.lax.with_sharding_constraint(expert_inputs, spec)
        expert_inputs = expert_inputs.reshape(self.n_experts, B * expert_capacity_per_batch, C) # n_experts, B * expert_cap, C

        expert_outputs = self.experts(expert_inputs) # n_experts, B * expert_cap, C
        if B % self.n_experts == 0:
            expert_outputs = expert_outputs.reshape(self.n_experts, -1, self.n_experts, expert_capacity_per_batch, C) # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_outputs = jnp.swapaxes(expert_outputs, 0, 2) # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_outputs = expert_outputs.reshape(B, self.n_experts, expert_capacity_per_batch, C) # B, n_experts, expert_cap, C
        else:
            expert_outputs = expert_outputs.reshape(self.n_experts, B, expert_capacity_per_batch, C) # n_experts, B, expert_cap, C
            expert_outputs = jnp.swapaxes(expert_outputs, 0, 1) # B, n_experts, expert_cap, C

        expert_outputs = jax.lax.with_sharding_constraint(expert_outputs, spec)

        y = jax.vmap(
            lambda x, i, p, prob: self._collect_outputs(x, i, p, prob)
            )(expert_outputs, expert_indices, expert_positions, top_k_probs)

        y = jax.lax.with_sharding_constraint(y, spec)

        if self.load_balance_loss is True:
            frac_tokens = jnp.bincount(expert_indices.flatten(), length=self.n_experts) / (2 * B * T) # distribution of tokens across experts
            frac_router_probs = jnp.sum(gate_probs, axis=(0, 1)) / (B * T) # distribution of gate probabilities across experts
            load_balance_loss = jnp.sum(frac_tokens * frac_router_probs) * self.n_experts
            output["load_balance_loss"] = load_balance_loss

        output["y"] = y
        return output

