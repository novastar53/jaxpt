from typing import Literal, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from jaxpt.modules.attention import CausalSelfAttention
from jaxpt.modules.mlp import MLP
from jaxpt.modules.config import Config
from jaxpt.utils import update_param, get_param

from transformers import GPT2LMHeadModel
import orbax.checkpoint as ocp


@dataclass(eq=True, unsafe_hash=True)
class GPTConfig(Config):
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
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
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
        self.attn = CausalSelfAttention(config, rngs=rngs)
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


class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
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
        self.wpe = nnx.Embed(
            config.block_size,
            config.n_embed,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=config.init_stddev, dtype=config.param_dtype),
                getattr(config, "pos_embed_partition_spec", (None,))
            ),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.h = [Block(config, rngs=rngs) for _ in range(config.n_layer)]
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
        T = idx.shape[1]
        pos = jnp.arange(0, T, dtype=jnp.uint16)
        pos_emb = self.wpe(pos)  # (T x C)
        tok_emb = self.wte(idx)  # (B x T x C)
        x = tok_emb + pos_emb
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
    def from_checkpoint(fpath: str, rngs: nnx.Rngs, config=Optional[GPTConfig]):
        config = config if config else GPTConfig()
        model = GPT(config=config, rngs=rngs)
        _, _, other_state = nnx.split(model, nnx.RngState, ...)
        checkpointer = ocp.StandardCheckpointer()
        other_state = checkpointer.restore(fpath, target=other_state)
        nnx.update(model, other_state)
        return model


def from_huggingface_pretrained(rngs: nnx.Rngs) -> GPT:
    import torch
    config = GPTConfig()
    model = GPT(config, rngs)
    graphdef, sd = nnx.split(model)

    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict()

    hf_keys = [k for k in sd_hf]
    transposed = ["lm_head.weight"]

    # assert len(sd_hf) == count_params(sd)

    for k in hf_keys:
        # map pytorch keys to flax keys
        jax_k = k
        if "lm_head" in jax_k:
            break
        if "transformer" in jax_k:
            jax_k = jax_k.replace("transformer.", "")

        if "wte" in jax_k or "wpe" in jax_k:
            jax_k = jax_k.replace("weight", "embedding")
        elif "ln_" in jax_k:
            jax_k = jax_k.replace("weight", "scale")
        else:
            jax_k = jax_k.replace("weight", "kernel")

        with torch.no_grad():
            hf_param = sd_hf[k].detach().cpu().numpy()
            jax_param = get_param(sd, jax_k).value
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                sd = update_param(sd, jax_k, jnp.array(hf_param).T)
                # check that the value was copied correctly
                test_param = get_param(sd, jax_k).value
                assert jnp.sum(test_param) == jnp.sum(hf_param.T)
                assert jnp.sum(test_param) != jnp.sum(jax_param)
                model = nnx.merge(graphdef, sd)

            else:
                # vanilla copy over the other parameters
                sd = update_param(sd, jax_k, jnp.array(hf_param))
                # check that the value was copied correctly
                test_param = get_param(sd, jax_k)
                assert jnp.sum(test_param.value) == jnp.sum(hf_param)
                assert jnp.sum(test_param.value) != jnp.sum(jax_param)
                model = nnx.merge(graphdef, sd)

    return model
