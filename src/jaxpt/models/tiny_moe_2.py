from typing import Literal, Optional
from dataclasses import dataclass, replace

import jax
import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp

from jaxpt.modules.config import Config
from jaxpt.modules.attention import GQ_Attention_w_RoPE
from jaxpt.modules.mlp import GLU, MLP
from jaxpt.modules.moe import MOE
from jaxpt.modules.position import (
    calc_rope_omega_llama,
)


@dataclass(eq=True, unsafe_hash=True)
class Tiny_MoE_2_Config(Config):
    name: str = "Tiny_MoE_2"
    dtype: jnp.dtype = jnp.bfloat16 # computation dype
    param_dtype: jnp.dtype = jnp.float32 # parameter dtype
    block_size: int = 2048  # sequence length
    vocab_size: int = 50304  # 50257 padded to the nearest multiple of 64
    n_layer: int = 30  # number of attention blocks
    n_head: int = 12  # number of attention heads
    n_kv_head: int = 4  # number of key-value heads
    n_embed: int = 672  # number token embedding dimensions
    n_experts: int = 8  # number of experts
    mesh: jax.sharding.Mesh | None = None  # device mesh
    top_k: int = 2  # number of top experts to use
    load_factor: float = 1.25 # load factor for expert buffers
    expert_weight_priority: bool = False # sort expert buffer assignments by expert weight
    load_balance_loss_coeff: float = 1e-2  # moe auxiliary loss coefficient
    z_loss_coeff: float = 5e-4 # moe z loss coefficient
    n_mlp_hidden: int = 2048  # number of hidden dimensions
    mlp_bias: bool = False  # use bias in mlp layers
    attention_bias: bool = False  # use bias in attention layers
    moe_bias: bool = False # use bias in moe layers
    moe_router_bias: bool = True # use bias in moe router
    ln_epsilon: float = 1e-5  # constant to prevent division by zero
    use_qk_norm: bool = True  # apply RMSNorm to Q and K before RoPE
    use_squared_relu: bool = True  # use squared ReLU activation in MoE experts
    sdpa_implementation: Literal["xla", "cudnn"] = (
        "xla"  # self-attention kernel implementation
    )
    rope_theta: int = int(1e-4)  # base frequency for rope
    init_stddev: float = 0.02  # stddev for layer init
    use_cache: bool = False  # use kv caching
    pad_token: str = "<pad>"

    glu_fc_kernel_sharding: tuple = (None,)
    glu_fc_bias_sharding: tuple = (None,)
    glu_gate_kernel_sharding: tuple = (None,)
    glu_gate_bias_sharding: tuple = (None,)
    glu_proj_kernel_sharding: tuple = (None,)
    glu_proj_bias_sharding: tuple = (None,)

    attn_wq_kernel_sharding: tuple = (None,)
    attn_wq_bias_sharding: tuple = (None,)
    attn_wkv_kernel_sharding: tuple = (None,)
    attn_wkv_bias_sharding: tuple = (None,)
    attn_wproj_kernel_sharding: tuple = (None,)
    attn_wproj_bias_sharding: tuple = (None,)

    embed_partition_spec: tuple = (None,)
    rmsnorm_partition_spec: tuple = (None,)



class Block(nnx.Module):
    def __init__(
        self, config: Tiny_MoE_2_Config, rope_omega: nnx.Variable, rngs: nnx.Rngs
    ) -> None:
        self.rms_n_1 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones,
                getattr(config, "rmsnorm_partition_spec", (None,))
            ),
            rngs=rngs,
        )
        self.attn = GQ_Attention_w_RoPE(
            config,
            rope_omega=rope_omega,
            rngs=rngs,
            use_qk_norm=getattr(config, "use_qk_norm", False),
        )
        self.rms_n_2 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones,
                getattr(config, "rmsnorm_partition_spec", (None,))
            ),
            rngs=rngs,
        )
        if config.moe_router_bias is True:
            moe_config = replace(config, mlp_bias=True)
        self.moe = MOE(moe_config, rngs)
        self.aux_loss = False

    def __call__(self, x, mask=None):
        x = self.attn(self.rms_n_1(x), mask=mask) + x
        output = self.moe(self.rms_n_2(x))
        x = x + output["y"] 
        output["y"] = x
        return output


class Tiny_MoE_2(nnx.Module):
    def __init__(self, config: Tiny_MoE_2_Config, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(
            config.vocab_size,
            config.n_embed,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=config.init_stddev,
                dtype=config.param_dtype),
                getattr(config, "embed_partition_spec", (None,))
            ),
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

        # pre-calculate the RoPE thetas
        omega = calc_rope_omega_llama(
            config.n_embed,
            config.n_head,
            config.block_size,
            config.rope_theta,
            config.dtype,
        )
        self.h = [ Block(config, rope_omega=omega, rngs=rngs) for _ in range(config.n_layer)]

        self.rms_n_f = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones,
                getattr(config, "rmsnorm_partition_spec", (None,))
            ),
            rngs=rngs,
        )
        self.n_layer = config.n_layer
        self.load_balance_loss = False
        self.z_loss = False


    def __call__(self, idx, mask=None):
        x = self.wte(idx)
        total_load_balance_loss = 0
        total_z_loss = 0
        for i in range(self.n_layer):
            output = self.h[i](x, mask)
            x = output["y"]
            if self.load_balance_loss:
                total_load_balance_loss += output["load_balance_loss"]
            if self.z_loss:
                total_z_loss += output["z_loss"]
        x = self.rms_n_f(x)
        logits = self.wte.attend(x)
        return logits, total_load_balance_loss, total_z_loss
    
            
    def save_checkpoint(self, fpath: str):
        _, _, other_state = nnx.split(self, nnx.RngState, ...)
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(fpath, other_state)
        ckptr.wait_until_finished()


    @staticmethod
    def from_checkpoint(
        fpath: str, rngs: nnx.Rngs, config: Optional[Tiny_MoE_2_Config] = None, sharding: Optional[jax.sharding.NamedSharding] = None
    ):
    
        default = jax.random.key(1337)
        gate_noise = jax.random.key(42)
        rngs = nnx.Rngs(default=default, gate_noise=gate_noise)
        config = config if config else Tiny_MoE_2_Config()
        abstract_model = nnx.eval_shape( 
            lambda: Tiny_MoE_2(config=config, rngs=nnx.Rngs(default=default, gate_noise=gate_noise))
        )
        graphdef, rngstate, other_state = nnx.split(
            abstract_model, nnx.RngState, ...
        )
        #pspecs = nnx.get_partition_spec(other_state)
        #sharded_state = nnx.with_sharding_constraint(other_state, pspecs)
        checkpointer = ocp.StandardCheckpointer()
        other_state = checkpointer.restore(fpath, target=other_state)
        model = nnx.merge(graphdef, rngstate, other_state)
        for i in range(len(model.h)):
            if hasattr(model.h[i], "moe"):
                #model.h[i].moe.gate_noise_rngstream = rngs["gate_noise"].fork()
                model.h[i].moe.gate_noise_rngstream = rngs.gate_noise # TODO: Temporary fix for backward compatibility with jax 0.5.2
        return model
