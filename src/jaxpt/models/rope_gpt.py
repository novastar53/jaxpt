from typing import Literal, Optional
from dataclasses import dataclass

import jax.numpy as jnp
import flax.nnx as nnx

from jaxpt.modules import RoPEAttention, MLP, Config

import orbax.checkpoint as ocp


@dataclass
class RoPE_GPTConfig(Config):
    dtype: jnp.dtype = jnp.float32
    block_size: int = 1024  # sequence length
    vocab_size: int = 50304  # 50257 padded to the nearest multiple of 64
    n_layer: int = 12  # number of attention blocks
    n_head: int = 12  # number of attention heads
    n_embed: int = 768  # number token embedding dimensionsa
    ln_epsilon: float = 1e-5
    sdpa_implementation: Literal["xla", "cudnn"] = "xla"
    apply_rope: bool = True



class Block(nnx.Module):
    def __init__(self, config: RoPE_GPTConfig, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(
            config.n_embed, epsilon=config.ln_epsilon, dtype=config.dtype, rngs=rngs
        )
        self.attn = RoPEAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(
            config.n_embed, epsilon=config.ln_epsilon, dtype=config.dtype, rngs=rngs
        )
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x


class RoPE_GPT(nnx.Module):
    def __init__(self, config: RoPE_GPTConfig, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(
            config.vocab_size,
            config.n_embed,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            dtype=config.dtype,
            param_dtype=config.dtype,
            rngs=rngs,
        )
        self.h = [Block(config, rngs=rngs) for _ in range(config.n_layer)]
        self.ln_f = nnx.LayerNorm(
            config.n_embed, epsilon=config.ln_epsilon, dtype=config.dtype, rngs=rngs
        )

    def __call__(self, idx):
        x = self.wte(idx)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.wte.attend(x)  # (B x T x V)
        return logits

    def save_checkpoint(self, fpath: str):
        _, _, other_state = nnx.split(self, nnx.RngState, ...)
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(fpath, other_state)

    @staticmethod
    def from_checkpoint(fpath: str, rngs: nnx.Rngs, config=Optional[RoPE_GPTConfig]):
        config = config if config else RoPE_GPTConfig()
        model = RoPE_GPT(config=config, rngs=rngs)
        _, _, other_state = nnx.split(model, nnx.RngState, ...)
        checkpointer = ocp.StandardCheckpointer()
        other_state = checkpointer.restore(fpath, target=other_state)
        nnx.update(model, other_state)
        return model


