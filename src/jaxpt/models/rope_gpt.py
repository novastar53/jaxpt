from typing import Literal, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from jaxpt.modules.attention import CausalSelfAttention_w_RoPE
from jaxpt.modules.mlp import MLP
from jaxpt.modules.config import Config
from jaxpt.utils import update_param, get_param
from jaxpt.modules.position import (
    calc_rope_omega_llama,
    calc_rope_omega_classic,
    RoPE_Llama,
    RoPE_Classic,
)
import orbax.checkpoint as ocp


@dataclass(eq=True, unsafe_hash=True)
class RoPE_GPTConfig(Config):
    dtype: jnp.dtype = jnp.float32
    block_size: int = 1024  # sequence length
    vocab_size: int = 50304  # 50257 padded to the nearest multiple of 64
    n_layer: int = 12  # number of attention blocks
    n_head: int = 12  # number of attention heads
    n_embed: int = 768  # number token embedding dimensions
    n_mlp_hidden: int = 4 * 768  # hidden size for piecewise FFN
    ln_epsilon: float = 1e-5
    init_stddev: float = 0.02  # stddev for layer init
    sdpa_implementation: Literal["xla", "cudnn"] = ("xla")
    rope_theta: float = 1e-4  # base frequency for rope
    
    #mesh: jax.sharding.Mesh | None = None  # device mesh
    
    embed_partition_spec: tuple = (None,)
    pos_embed_partition_spec: tuple = (None,)
    ln_partition_spec: tuple = (None,)

    mlp_fc_kernel_sharding: tuple = (None,)
    mlp_fc_bias_sharding: tuple = (None,)
    mlp_proj_kernel_sharding: tuple = (None,)
    mlp_proj_bias_sharding: tuple = (None,)

    attn_c_attn_kernel_sharding: tuple = (None,)
    attn_c_attn_bias_sharding: tuple = (None,)
    attn_c_proj_kernel_sharding: tuple = (None,)
    attn_c_proj_bias_sharding: tuple = (None,)


class Block(nnx.Module):
    def __init__(self, config: RoPE_GPTConfig, rope_omega: nnx.Variable, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            dtype=config.dtype,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones,
                getattr(config, "ln_partition_spec", (None,))
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                getattr(config, "ln_partition_spec", (None,))
            ),
            rngs=rngs,
        )
        self.attn = CausalSelfAttention_w_RoPE(config, rope_omega=rope_omega, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            dtype=config.dtype,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones,
                getattr(config, "ln_partition_spec", (None,))
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                getattr(config, "ln_partition_spec", (None,))
            ),
            rngs=rngs,
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
            embedding_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=config.init_stddev, dtype=config.param_dtype),
                getattr(config, "embed_partition_spec", (None,))
            ),
            dtype=config.dtype,
            param_dtype=config.dtype,
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
        self.h = [Block(config, rope_omega=omega, rngs=rngs) for _ in range(config.n_layer)]
        self.ln_f = nnx.LayerNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            dtype=config.dtype,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones,
                getattr(config, "ln_partition_spec", (None,))
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                getattr(config, "ln_partition_spec", (None,))
            ),
            rngs=rngs,
        )

    def __call__(self, idx):
        x = self.wte(idx)  # (B x T x C)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.wte.attend(x)  # (B x T x V)
        return logits

    def save_checkpoint(self, fpath: str):
        _, _, other_state = nnx.split(self, nnx.RngState, ...)
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(fpath, other_state)
        ckptr.wait_until_finished()

    @staticmethod
    def from_checkpoint(fpath: str, rngs: nnx.Rngs, config=Optional[RoPE_GPTConfig]):
        config = config if config else RoPE_GPTConfig()
        model = RoPE_GPT(config=config, rngs=rngs)
        _, _, other_state = nnx.split(model, nnx.RngState, ...)
        checkpointer = ocp.StandardCheckpointer()
        other_state = checkpointer.restore(fpath, target=other_state)
        nnx.update(model, other_state)
        return model
