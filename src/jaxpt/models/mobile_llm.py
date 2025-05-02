from typing import Literal, Optional
from dataclasses import dataclass

import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp

from jaxpt.modules.config import Config
from jaxpt.modules.attention import MQ_Attention, RoPEAttention
from jaxpt.modules.mlp import GLU, MLP
from jaxpt.modules.position import calc_rope_omega

@dataclass
class MobileLLM_Config(Config):
    dtype: jnp.dtype = jnp.float32
    block_size: int = 2048  # sequence length
    vocab_size: int = 50257  # 50257 padded to the nearest multiple of 64
    n_layer: int = 30 # number of attention blocks
    n_head: int = 9 # number of attention heads
    n_kv_head: int = 3 # number of shared key-value heads
    n_embed: int = 576  # number token embedding dimensionsa
    n_mlp_hidden: int = 1536 # number of hidden dimensions
    glu_activation: Literal["gelu", "silu", "sigmoid"] = "silu" # glu activation or gating function
    ln_epsilon: float = 1e-5 # constant to prevent division by zero
    sdpa_implementation: Literal["xla", "cudnn"] = "xla" # self-attention kernel implementation
    rope_theta: int = 1e-5 # base frequency for rope
    init_stddev: float = 0.02 # stddev for layer init


class Block(nnx.Module):
    def __init__(self, config: MobileLLM_Config, rope_omega: nnx.Variable, 
                 rngs: nnx.Rngs) -> None:
        self.ln_1 = nnx.LayerNorm(
            config.n_embed, epsilon=config.ln_epsilon, 
            dtype=config.dtype, rngs=rngs
        )
        self.attn = MQ_Attention(
            config, rope_omega=rope_omega, rngs=rngs
        )
        self.ln_2 = nnx.LayerNorm(
            config.n_embed, epsilon=config.ln_epsilon,
            dtype=config.dtype, rngs=rngs
        )
        self.mlp = GLU(config, rngs)
    
    def __call__(self, x):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x


class Mobile_LLM(nnx.Module):
    def __init__(self, config: MobileLLM_Config, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(
            config.vocab_size,
            config.n_embed,
            embedding_init=nnx.initializers.normal(stddev=config.init_stddev),
            rngs=rngs
        )

        # pre-calculate the RoPE thetas
        omega = calc_rope_omega(config.n_embed, 
                                config.n_head, 
                                config.block_size,
                                config.rope_theta, 
                                config.dtype)
        self.h = [Block(config, rope_omega=omega, rngs=rngs) 
                  for _ in range(config.n_layer)]

        self.ln_f = nnx.LayerNorm(
            config.n_embed, epsilon=config.ln_epsilon, 
            dtype=config.dtype, rngs=rngs
        )

    def __call__(self, idx):
        x = self.wte(idx)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.wte.attend(x)
        return logits

    def save_checkpoint(self, fpath: str):
        _, _, other_state = nnx.split(self, nnx.RngState, ...)
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(fpath, other_state)

    @staticmethod
    def from_checkpoint(fpath: str, rngs: nnx.Rngs, config=Optional[MobileLLM_Config]):
        config = config if config else MobileLLM_Config()
        model = Mobile_LLM(config=config, rngs=rngs)
        _, _, other_state = nnx.split(model, nnx.RngState, ...)
        checkpointer = ocp.StandardCheckpointer()
        other_state = checkpointer.restore(fpath, target=other_state)
        nnx.update(model, other_state)
