from typing import Literal, Optional
from dataclasses import dataclass

import jax
import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp

from jaxpt.modules.config import Config
from jaxpt.modules.attention import GQ_Attention
from jaxpt.modules.mlp import GLU, MLP
from jaxpt.modules.position import (
    calc_rope_omega_llama,
    calc_rope_omega_classic,
    RoPE_Llama,
    RoPE_Classic,
)
from jaxpt.utils import create_sharded_model


@dataclass(eq=True, unsafe_hash=True)
class SmolLM_Config(Config):
    name: str = "SmolLM"
    dtype: jnp.dtype = jnp.float32
    block_size: int = 2048  # sequence length
    vocab_size: int = 50257  # 50257 padded to the nearest multiple of 64
    n_layer: int = 30  # number of attention blocks
    n_head: int = 9  # number of attention heads
    n_kv_head: int = 3  # number of key-value heads
    n_embed: int = 576  # number token embedding dimensionsa
    n_mlp_hidden: int = 1536  # number of hidden dimensions
    mlp_bias: bool = False  # use bias in mlp layers
    attention_bias: bool = False  # use bias in attention layers
    glu_activation: Literal["gelu", "silu", "sigmoid"] = (
        "silu"  # glu activation or gating function
    )
    ln_epsilon: float = 1e-5  # constant to prevent division by zero
    sdpa_implementation: Literal["xla", "cudnn"] = (
        "xla"  # self-attention kernel implementation
    )
    rope_theta: int = 1e-4  # base frequency for rope
    init_stddev: float = 0.02  # stddev for layer init
    use_cache: bool = False  # use kv caching
    pad_token: str = "<pad>"


class Block(nnx.Module):
    def __init__(
        self, config: SmolLM_Config, rope_omega: nnx.Variable, rngs: nnx.Rngs
    ) -> None:
        self.rms_n_1 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            scale_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                ("model",)
            ),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.attn = GQ_Attention(config, rope_omega=rope_omega, rngs=rngs)
        self.rms_n_2 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            scale_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                ("model",)
            ),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.mlp = GLU(config, rngs)

    def __call__(self, x, mask=None):
        x = self.attn(self.rms_n_1(x), mask=mask) + x
        x = self.mlp(self.rms_n_2(x)) + x
        return x


class SmolLM(nnx.Module):
    def __init__(self, config: SmolLM_Config, rngs: nnx.Rngs):
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
        self.h = [
            Block(config, rope_omega=omega, rngs=rngs)
            for _ in range(config.n_layer)
        ]

        self.rms_n_f = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones,
                ("model",)
            ),
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
        fpath: str, rngs: nnx.Rngs, config=Optional[SmolLM_Config]
    ):
        config = config if config else SmolLM_Config()
        model = SmolLM(config=config, rngs=rngs)
        _, _, other_state = nnx.split(model, nnx.RngState, ...)
        checkpointer = ocp.StandardCheckpointer()
        other_state = checkpointer.restore(fpath, target=other_state)
        nnx.update(model, other_state)
        return model


def from_hf_pretrained(config: SmolLM_Config, rngs: nnx.Rngs, sharded=False) -> SmolLM:
    if sharded:
        m = SmolLM(config, rngs)
    else:
        m = create_sharded_model(SmolLM, config, rngs)

    graphdef, flax_params, other_state = nnx.split(m, nnx.Param, ...)

    hf_m = load_hf_pretrained(model_name=config.name)
    state = hf_m.state_dict()

    flax_params.wte.embedding.value = jnp.array(
        state["model.embed_tokens.weight"].numpy(), dtype=config.dtype
    )
    flax_params.rms_n_f.scale.value = jnp.array(
        state["model.norm.weight"].numpy(), dtype=config.dtype
    )

    for i in range(len(flax_params.h)):
        # MLP weights
        flax_params.h[i].mlp.gate.kernel.value = jnp.array(
            state[f"model.layers.{i}.mlp.gate_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        flax_params.h[i].mlp.c_fc.kernel.value = jnp.array(
            state[f"model.layers.{i}.mlp.up_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        flax_params.h[i].mlp.c_proj.kernel.value = jnp.array(
            state[f"model.layers.{i}.mlp.down_proj.weight"].numpy().T,
            dtype=config.dtype,
        )

        # RMS Norm weights
        flax_params.h[i].rms_n_1.scale.value = jnp.array(
            state[f"model.layers.{i}.input_layernorm.weight"].numpy(),
            dtype=config.dtype,
        )
        flax_params.h[i].rms_n_2.scale.value = jnp.array(
            state[f"model.layers.{i}.post_attention_layernorm.weight"].numpy(),
            dtype=config.dtype,
        )

        # Causal self-attention weights
        flax_params.h[i].attn.wproj.kernel.value = jnp.array(
            state[f"model.layers.{i}.self_attn.o_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        flax_params.h[i].attn.wq.kernel.value = jnp.array(
            state[f"model.layers.{i}.self_attn.q_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        wk = jnp.array(
            state[f"model.layers.{i}.self_attn.k_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        wv = jnp.array(
            state[f"model.layers.{i}.self_attn.v_proj.weight"].numpy().T,
            dtype=config.dtype,
        )
        wkv = jnp.concatenate([wk, wv], axis=1)
        flax_params.h[i].attn.wkv.kernel.value = jnp.array(
            wkv, dtype=config.dtype
        )

    m = nnx.merge(graphdef, flax_params, other_state)

    return m


def load_hf_pretrained(model_name="HuggingFaceTB/SmolLM-135M"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model


def convert_to_hf(m: SmolLM):
    import torch
    import numpy as np

    _, flax_params, _ = nnx.split(m, nnx.Param, ...)
    hf_smol_llm = load_hf_pretrained()

    state = hf_smol_llm.state_dict()

    state["model.embed_tokens.weight"] = torch.from_numpy(
        np.array(flax_params.wte.embedding.value)
    )
    state["model.norm.weight"] = torch.from_numpy(
        np.array(flax_params.rms_n_f.scale.value)
    )

    for i in range(len(flax_params.h)):
        # MLP weights
        state[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.from_numpy(
            np.array(flax_params.h[i].mlp.gate.kernel.value).T
        )
        state[f"model.layers.{i}.mlp.up_proj.weight"] = torch.from_numpy(
            np.array(flax_params.h[i].mlp.c_fc.kernel.value).T
        )
        state[f"model.layers.{i}.mlp.down_proj.weight"] = torch.from_numpy(
            np.array(flax_params.h[i].mlp.c_proj.kernel.value).T
        )

        # RMS Norm weights
        state[f"model.layers.{i}.input_layernorm.weight"] = torch.from_numpy(
            np.array(flax_params.h[i].rms_n_1.scale.value)
        )
        state[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            torch.from_numpy(np.array(flax_params.h[i].rms_n_2.scale.value))
        )

        # Causal self-attention weights
        len_k = m.config.n_kv_head * m.config.n_embed // m.config.n_head
        state[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.from_numpy(
            np.array(flax_params.h[i].attn.wproj.kernel.value)
        )
        state[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.from_numpy(
            np.array(flax_params.h[i].attn.wq.kernel.value)
        )
        wk = flax_params.h[i].attn.wkv.kernel.value[:, :len_k]
        state[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.from_numpy(
            np.array(wk.T)
        )
        wv = flax_params.h[i].attn.wkv.kernel.value[:, len_k:]
        state[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.from_numpy(
            np.array(wv.T)
        )

    hf_smol_llm.load_state_dict(state, strict=True)

    return hf_smol_llm
