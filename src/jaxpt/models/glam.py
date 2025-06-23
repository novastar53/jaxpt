from typing import Literal, Optional
from dataclasses import dataclass

import jax
import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp

from jaxpt.modules.config import Config
from jaxpt.modules.attention import GQ_Attention
from jaxpt.modules.mlp import GLU, MLP, MOE
from jaxpt.modules.position import (
    calc_rope_omega_llama,
    calc_rope_omega_classic,
    RoPE_Llama,
    RoPE_Classic,
)


@dataclass(eq=True, unsafe_hash=True)
class GLaM_Config(Config):
    name: str = "GLaM"
    dtype: jnp.dtype = jnp.float32
    block_size: int = 1024  # sequence length
    vocab_size: int = 50304 # 50257 padded to the nearest multiple of 64
    n_layer: int = 12  # number of attention blocks
    n_head: int = 12  # number of attention heads
    n_kv_head: int = 4  # number of key-value heads
    n_embed: int = 768  # number token embedding dimensionsa
    n_experts: int = 64  # number of experts
    n_top_k_experts: int = 2  # number of top experts to use
    aux_loss_coeff: float = 1e-2 # moe auxiliary loss coefficient
    n_mlp_hidden: int = 3072  # number of hidden dimensions
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


class GLU_Block(nnx.Module):
    def __init__(
        self, config: GLaM_Config, rope_omega: nnx.Variable, rngs: nnx.Rngs
    ) -> None:
        self.rms_n_1 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.attn = GQ_Attention(config, rope_omega=rope_omega, rngs=rngs)
        self.rms_n_2 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.mlp = GLU(config, rngs)

    def __call__(self, x, mask=None):
        x = self.attn(self.rms_n_1(x), mask=mask) + x
        x = self.mlp(self.rms_n_2(x)) + x
        return x


class MOE_Block(nnx.Module):
    def __init__(
        self, config: GLaM_Config, rope_omega: nnx.Variable, rngs: nnx.Rngs
    ) -> None:
        self.rms_n_1 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.attn = GQ_Attention(config, rope_omega=rope_omega, rngs=rngs)
        self.rms_n_2 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.mlp = MOE(config, rngs)

    def __call__(self, x, mask=None):
        x = self.attn(self.rms_n_1(x), mask=mask) + x
        x = self.mlp(self.rms_n_2(x)) + x
        return x


class GLaM(nnx.Module):
    def __init__(self, config: GLaM_Config, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(
            config.vocab_size,
            config.n_embed,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=config.init_stddev),
                (None, "model")),
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
        for _ in range(config.n_layer//2):
            self.h += [
                MOE_Block(config, rope_omega=omega, rngs=rngs),
                GLU_Block(config, rope_omega=omega, rngs=rngs),
            ]

        self.rms_n_f = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            dtype=config.dtype,
            rngs=rngs,
        )

    def __call__(self, idx, mask=None):
        x = self.wte(idx)
        for block in self.h:
            x = block(x, mask)
        x = self.rms_n_f(x)
        logits = self.wte.attend(x)
        return logits

    def save_checkpoint(self, fpath: str):
        _, _, other_state = nnx.split(self, nnx.RngState, ...)
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(fpath, other_state)

    @staticmethod
    def from_checkpoint(
        fpath: str, rngs: nnx.Rngs, config: Optional[GLaM_Config]
    ):
        config = config if config else GLaM_Config()
        model = GLaM(config=config, rngs=rngs)
        abstract_model = nnx.eval_shape(lambda: GLaM(config=config, rngs=rngs))
        graphdef, rngstate, other_state = nnx.split(abstract_model, nnx.RngState, ...)
        checkpointer = ocp.StandardCheckpointer()
        other_state = checkpointer.restore(fpath, target=other_state)
        model = nnx.merge(graphdef, rngstate, other_state)
        return model

