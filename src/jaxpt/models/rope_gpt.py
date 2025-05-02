from typing import Literal, Optional
from dataclasses import dataclass

import jax.numpy as jnp
import flax.nnx as nnx

from jaxpt.modules.attention import RoPEAttention
from jaxpt.modules.mlp import MLP
from jaxpt.modules.config import Config

import orbax.checkpoint as ocp


@dataclass
class RoPE_GPTConfig(Config):
    dtype: jnp.dtype = jnp.float32
    block_size: int = 1024  # sequence length
    vocab_size: int = 50304  # 50257 padded to the nearest multiple of 64
    n_layer: int = 12  # number of attention blocks
    n_head: int = 12  # number of attention heads
    n_embed: int = 768  # number token embedding dimensionsa
    n_mlp_hidden: int = 4 * 768 # hiden size for piecewise FFN
    ln_epsilon: float = 1e-5
    sdpa_implementation: Literal["xla", "cudnn", "slow"] = "xla"
    rope_base_freq: float = 1e-5



class Block(nnx.Module):
    def __init__(self, config: RoPE_GPTConfig, omega: nnx.Variable, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(
            config.n_embed, epsilon=config.ln_epsilon, dtype=config.dtype, rngs=rngs
        )
        self.attn = RoPEAttention(config, omega, rngs=rngs)
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

        # pre-calculate the RoPE thetas
        query_size = config.n_embed // config.n_head
        base_freq = config.rope_base_freq**(2/query_size)
        omega = jnp.ones((1, query_size), dtype=config.dtype) * base_freq
        pow = jnp.arange(0, query_size)
        omega = jnp.repeat(omega**pow, config.block_size, axis=0)
        pos = jnp.arange(0, config.block_size)
        pos = jnp.expand_dims(pos, axis=1)
        omega = omega * pos
        omega = nnx.Variable(omega)
        self.h = [Block(config, omega, rngs=rngs) for _ in range(config.n_layer)]

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


