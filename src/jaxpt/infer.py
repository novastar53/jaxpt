import sys
from functools import partial
from typing import Literal
import logging

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import tiktoken

from jaxpt.chatml import format_as_gpt4_chatml_and_tokenize


logger = logging.getLogger(__name__)

@nnx.jit
def _generate_step(m, x, attn_mask, temperature, key):
    logits = m(x, mask=attn_mask) / temperature
    x_new = logits[:, -1, :]
    top_k_vals, top_k_indices = jax.lax.top_k(x_new, 50)
    key, subkey = jax.random.split(key)
    top_k_logit_idxs = jax.random.categorical(
        subkey, top_k_vals
    )
    top_k_logit_idxs = top_k_logit_idxs[..., None]  # expand dims
    sample_idxs = jnp.take_along_axis(
        top_k_indices, top_k_logit_idxs, axis=-1
    )
    return sample_idxs


def generate_completion_slow(
    model,
    enc=None,
    prefix="Hello, I'm a language model,",
    attn_mask: jax.Array | None = None,
    num_completions=8,
    max_length=20,
    temperature=0.2,
    key=None,
):
    if enc is None:
        enc = tiktoken.get_encoding("gpt2")
    if key is None:
        key = jax.random.PRNGKey(1337)

    tokens = enc.encode(prefix)
    tokens = jnp.array(tokens, dtype=jnp.int32)
    tokens = jnp.expand_dims(tokens, axis=0)
    x = jnp.tile(tokens, (num_completions, 1))
    while x.shape[1] < max_length:
        x_next = _generate_step(model, x, attn_mask, temperature, key)
        x = jnp.concatenate((x, x_next), axis=1)  # (B, T+1)#
        if attn_mask:
            attn_mask = jnp.concatenate((attn_mask, 
                                        jnp.ones((attn_mask.shape[0], 1), dtype=jnp.bool)), axis=1)
    output = []
    for i in range(num_completions):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        output.append(decoded)
    return output


def generate_completion_fast(
    model,
    enc=None,
    prefix="Hello, I'm a language model,",
    attn_mask: jax.Array | None = None,
    num_completions=8,
    max_length=20,
    temperature=0.2,
    key=None,
):
    if enc is None:
        enc = tiktoken.get_encoding("gpt2")
    if key is None:
        key = jax.random.PRNGKey(1337)

    tokens = enc.encode(prefix)
    tokens = jnp.array(tokens, dtype=jnp.int32)
    tokens = jnp.expand_dims(tokens, axis=0)
    x = jnp.tile(tokens, (num_completions, 1))
    #attn_mask = jnp.ones((attn_mask.shape[0], 1), dtype=jnp.bool)
    while x.shape[1] < max_length:
        x_next = _generate_step(model, x, attn_mask, temperature, key)
        x = x_next
    output = []
    for i in range(num_completions):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        output.append(decoded)
    return output


def generate_chat(
    model,
    x_prev=None,
    enc=None,
    system_prompt="You are a helpful assistant.",
    question="What is photosynthesis?",
    format: Literal["chatml", "completion"] = "completion",
    stop_tokens=("<|endoftext|>", "<|im_end|>"),
    temperature=0.2,
    top_k=50,
    key=None,
    logger=logger,
):
    stop_tokens = set([enc.encode(s)[0] for s in stop_tokens])

    if enc is None:
        enc = tiktoken.get_encoding("gpt2")
    if key is None:
        key = jax.random.PRNGKey(1337)

    match format:
        case "chatml":
            x = format_as_gpt4_chatml_and_tokenize(
                tokenizer=enc,
                system_prompt=system_prompt,
                question=question,
                start=(x_prev is None),
                logger=logger,
            )
        case "completion" | _:
            x = jnp.array(enc.encode(question))

    x = x[None, ...]
    if x_prev is not None:
        x = jnp.concatenate((x_prev, x), axis=1)  # (B, T+1)#
    logger.debug(f"Context length: {x.shape[1]}")
    print("Model: ", end="")
    try:
        while True:
            logits = model(x)[:, -1, :] / temperature
            key, subkey = jax.random.split(key)
            x_next, key = top_k_sampling(logits, subkey, k=top_k)
            if int(x_next[0]) in stop_tokens:
                break
            decoded = enc.decode(x_next)
            print(decoded, end="")
            sys.stdout.flush()
            x_next = x_next[..., None]
            x = jnp.concatenate((x, x_next), axis=1)  # (B, T+1)#
    except KeyboardInterrupt:
        pass
    finally:
        print("\n")
    return x
