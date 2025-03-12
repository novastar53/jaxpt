from typing import Literal
from dataclasses import dataclass
from functools import partial

import torch
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.experimental import mesh_utils

from jaxpt.utils import count_params, update_param, get_param

from transformers import GPT2LMHeadModel
from flash_attention_jax import causal_flash_attention
#import jax.experimental.pallas.ops.gpu.attention as pallas_attn
import orbax.checkpoint as ocp



@dataclass
class GPTConfig:
    dtype: jnp.dtype = jnp.float32
    block_size: int = 1024  # sequence length
    vocab_size: int = 50304  # 50257 padded to the nearest multiple of 64
    n_layer: int = 12  # number of attention blocks
    n_head: int = 12  # number of attention heads
    n_embed: int = 768  # number token embedding dimensionsa
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    ln_epsilon: float = 1e-5
    sdpa_implementation: Literal["xla", "cudnn"] = "xla"

#@partial(jax.jit, static_argnames=("approximate",))
#def mygelu(x, approximate=True):
#    return nnx.gelu(x, approximate=approximate)

class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.c_attn = nnx.Linear(
            config.n_embed,
            3 * config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init=nnx.initializers.normal(
                stddev=0.02 * (2 * config.n_layer) ** -0.5
            ),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.mask = nnx.Variable(
            jnp.tril(
                jnp.ones(
                    (config.block_size, config.block_size), dtype=config.dtype
                ).reshape((1, 1, config.block_size, config.block_size))
            )
        )
        
        #self.attn_dropout = nnx.Dropout(config.attn_pdrop, rngs=rngs)
        self.resid_dropout = nnx.Dropout(config.resid_pdrop, rngs=rngs)
        #self.attn = partial(nnx.dot_product_attention, dropout_rate=config.attn_pdrop, dropout_rng=rngs.dropout.key.value)
        #self.rngs = rngs
        self.implementation = config.sdpa_implementation

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)  # B, T, 3 * C
        q, k, v = jnp.split(qkv, 3, axis=-1)  # 3 * (B, T, C)

        q = jnp.reshape(
            q, (B, T, self.n_head, C // self.n_head)
        )  # B, T, n_head, C // n_head
        #q = jnp.transpose(q, axes=(0, 2, 1, 3)) # B, n_head, T, C // n_head

        k = jnp.reshape(
            k, (B, T, self.n_head, C // self.n_head)
        )  # B, T, n_head, C // n_head
        #k = jnp.transpose(k, axes=(0, 2, 1, 3)) # B, n_head, T, C // n_head

        v = jnp.reshape(
            v, (B, T, self.n_head, C // self.n_head)
        )  # B, T, n_head, C // n_head
        #v = jnp.transpose(v, axes=(0, 2, 1, 3)) # B, n_head, T, C // n_head

        #att = ( q @ jnp.transpose(k, axes=(0, 1, 3, 2))) / jnp.sqrt(k.shape[-1])  # B, n_head, T, T
        #att = jnp.where(self.mask[:, :, :T, :T] == 0.0, float('-inf'), att)
        #att = jax.nn.softmax(att, axis=-1)
        #att = self.attn_dropout(att)
        #y = att @ v # (B, n_head, T, T) x (b, n_head, T, hs) -> (B, n_head, T, hs)
        #y = jnp.transpose(y, axes=(0, 2, 1, 3)) # (B, T, n_head, hs)


        # alternative implementations
        #y = self.attn(query=q, key=k, value=v, mask=self.mask[:, :, :T, :T]) 
        #y = pallas_attn.mha(q, k, v, segment_ids=None, causal=True)
        #y = causal_flash_attention(q, k, v)
        #y = self.attn(q, k, v, mask=self.mask[:, :, :T, :T])

        # based on https://github.com/MasterSkepticista/gpt2/blob/5799d821b71c25d57f97159835a516689b3fe607/model.py
        # he hasn't used dropout in the attention weights. hopefully it won't affect accuracy
        # too much
        y = jax.nn.dot_product_attention(q, k, v, is_causal=True,
                                          implementation=self.implementation) 
        
        y = jnp.reshape(y, (B, T, C))  # (B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(
            config.n_embed,
            4 * config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            4 * config.n_embed,
            config.n_embed,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(config.resid_pdrop, rngs=rngs)

    def __call__(self, x):
        x = self.c_fc(x)
        x = nnx.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(config.n_embed, epsilon=config.ln_epsilon, dtype=config.dtype, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embed, epsilon=config.ln_epsilon, dtype=config.dtype, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x


class GPT2(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(
            config.vocab_size,
            config.n_embed,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            dtype=config.dtype,
            param_dtype=config.dtype,
            rngs=rngs,
        )
        self.wpe = nnx.Embed(
            config.block_size,
            config.n_embed,
            embedding_init=nnx.initializers.normal(stddev=0.02),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(config.embd_pdrop, rngs=rngs)
        self.h = [Block(config, rngs=rngs) for _ in range(config.n_layer)]
        self.ln_f = nnx.LayerNorm(config.n_embed, epsilon=config.ln_epsilon, dtype=config.dtype, rngs=rngs)

    def __call__(self, idx):
        T = idx.shape[1]
        pos = jnp.arange(0, T, dtype=jnp.uint16)
        pos_emb = self.wpe(pos)
        tok_emb = self.wte(idx)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.wte.attend(x) # (B x T x V)
        return logits


def save_checkpoint(model, fpath: str):
    _, _, other_state = nnx.split(model, nnx.RngState, ...)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(fpath, other_state)
    

def from_checkpoint(fpath: str, rngs: nnx.Rngs):
    checkpointer = ocp.StandardCheckpointer()
    model = GPT2(GPTConfig(), rngs=rngs)
    _, _, other_state = nnx.split(model, nnx.RngState, ...)
    other_state = checkpointer.restore(fpath, target=other_state) 
    nnx.update(model, other_state)
    return model

'''
    @classmethod
    def from_pretrained(cls, rngs: nnx.Rngs):
        config = GPTConfig()
        model = GPT2(config, rngs)
        graphdef, sd = nnx.split(model)

        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        sd_hf = model_hf.state_dict()

        hf_keys = [k for k in sd_hf]
        transposed = ["lm_head.weight"]

        assert len(sd_hf) == count_params(sd)

        for k in hf_keys:
            # map pytorch keys to flax keys
            if "wte" in k or "wpe" in k:
                jax_k = k.replace("weight", "embedding")
            elif "ln_" in k:
                jax_k = k.replace("weight", "scale")
            else:
                jax_k = k.replace("weight", "kernel")
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

        return model, model_hf
'''