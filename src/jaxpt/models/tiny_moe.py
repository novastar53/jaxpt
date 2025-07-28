from typing import Literal, Optional
from dataclasses import dataclass

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
    calc_rope_omega_classic,
    RoPE_Llama,
    RoPE_Classic,
)


@dataclass(eq=True, unsafe_hash=True)
class Tiny_MoE_Config(Config):
    name: str = "Tiny_MoE"
    dtype: jnp.dtype = jnp.bfloat16 # computation dype
    param_dtype: jnp.dtype = jnp.float32 # parameter dtype
    block_size: int = 2048  # sequence length
    vocab_size: int = 50304  # 50257 padded to the nearest multiple of 64
    n_layer: int = 32  # number of attention blocks
    n_head: int = 12  # number of attention heads
    n_kv_head: int = 4  # number of key-value heads
    n_embed: int = 576  # number token embedding dimensionsa
    n_experts: int = 8  # number of experts
    mesh: jax.sharding.Mesh = None  # device mesh
    top_k: int = 2  # number of top experts to use
    load_factor: int = 1.1 # load factor for expert buffers
    expert_weight_priority: bool = True # sort expert buffer assignments by expert weight 
    aux_loss_coeff: float = 1e-2  # moe auxiliary loss coefficient
    n_mlp_hidden: int = 2304  # number of hidden dimensions
    mlp_bias: bool = False  # use bias in mlp layers
    attention_bias: bool = False  # use bias in attention layers
    ln_epsilon: float = 1e-5  # constant to prevent division by zero
    glu_activation: Literal["sigmoid", "gelu", "silu"] = "silu"
    sdpa_implementation: Literal["xla", "cudnn"] = (
        "xla"  # self-attention kernel implementation
    )
    rope_theta: int = 1e-4  # base frequency for rope
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


class GLU_Block(nnx.Module):
    def __init__(
        self, config: Tiny_MoE_Config, rope_omega: nnx.Variable, rngs: nnx.Rngs
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
        self.attn = GQ_Attention_w_RoPE(config, rope_omega=rope_omega, rngs=rngs)
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
        self.glu = GLU(config, rngs)

    def __call__(self, x, mask=None):
        x = self.attn(self.rms_n_1(x), mask=mask) + x
        x = self.glu(self.rms_n_2(x)) + x
        return x


class MOE_Block(nnx.Module):
    def __init__(
        self, config: Tiny_MoE_Config, rope_omega: nnx.Variable, rngs: nnx.Rngs
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
        self.attn = GQ_Attention_w_RoPE(config, rope_omega=rope_omega, rngs=rngs)
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
        self.moe = MOE(config, rngs)
        self.aux_loss = False

    def __call__(self, x, mask=None):
        x = self.attn(self.rms_n_1(x), mask=mask) + x
        if self.aux_loss is True:
            moe_out, moe_aux_loss = self.moe(self.rms_n_2(x))
            x = moe_out + x
            return x, moe_aux_loss
        else:
            moe_out = self.moe(self.rms_n_2(x))
            x = moe_out + x
            return x


class Tiny_MoE(nnx.Module):
    def __init__(self, config: Tiny_MoE_Config, rngs: nnx.Rngs):
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
        self.h = []
        for _ in range(config.n_layer // 2):
            self.h += [
                MOE_Block(config, rope_omega=omega, rngs=rngs),
                GLU_Block(config, rope_omega=omega, rngs=rngs),
            ]

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
        self.aux_loss = False
        self.n_layer = config.n_layer

    def __call__(self, idx, mask=None):
        x = self.wte(idx)
        total_aux_loss = 0
        for i in range(0, self.n_layer, 2):
            if self.aux_loss is True:
                x, aux_loss = self.h[i](x, mask)
                total_aux_loss += aux_loss
            else:
                x = self.h[i](x, mask)
            x = self.h[i+1](x, mask)
        x = self.rms_n_f(x)
        logits = self.wte.attend(x)
        if self.aux_loss is True:
            return logits, total_aux_loss
        return logits
    
            
    def save_checkpoint(self, fpath: str):
        _, _, other_state = nnx.split(self, nnx.RngState, ...)
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(fpath, other_state)
        ckptr.wait_until_finished()

    @staticmethod
    def from_checkpoint(
        fpath: str, rngs: nnx.Rngs, config: Optional[Tiny_MoE_Config], sharding: Optional[jax.sharding.NamedSharding]
    ):
    
        default = jax.random.key(1337)
        gate_noise = jax.random.key(42)
        rngs = nnx.Rngs(default=default, gate_noise=gate_noise)
        config = config if config else Tiny_MoE_Config()
        abstract_model = nnx.eval_shape( 
            lambda: Tiny_MoE(config=config, rngs=nnx.Rngs(default=default, gate_noise=gate_noise))
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
                model.h[i].moe.gate_noise_rngstream = rngs["gate_noise"].fork()
        return model
