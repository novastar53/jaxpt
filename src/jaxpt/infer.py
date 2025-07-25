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


def top_k_sampling(logits, key, k=50):
    top_k_logits, top_k_indices = jax.lax.top_k(
        logits, k=k
    )  # Get the top k logits and indices
    key, subkey = jax.random.split(key)  # generate a new random key
    top_k_logit_idxs = jax.random.categorical(
        subkey, top_k_logits
    )  # sample from the top k logits
    top_k_logit_idxs = top_k_logit_idxs[..., None]  # expand dims
    sample_idxs = jnp.take_along_axis(
        top_k_indices, top_k_logit_idxs, axis=-1
    )  # pick out the sampled tokens
    sample_idxs = sample_idxs[..., 0]  # squeeze dims
    return sample_idxs, key


def generate_slow(
    model: nnx.Module,
    *,
    x: jax.Array,
    key: jax.random.PRNGKey,
    max_length=50,
    temperature=0.2,
    top_k=50,
) -> jax.Array:
    while x.shape[1] < max_length:
        logits = model(x) / temperature
        logits = logits[:, -1, :]
        x_next, key = top_k_sampling(logits, key, k=top_k)
        x_next = x_next[..., None]
        x = jnp.concatenate((x, x_next), axis=1)  # (B, T+1)#
    return x


def generate(
    model: nnx.Module,
    *,
    x: jax.Array,
    key: jax.random.PRNGKey,
    max_length=50,
    temperature=0.2,
    top_k=50,
) -> jax.Array:
    logits = model(x) / temperature
    logits = logits[:, -1, :]
    x_next, key = top_k_sampling(logits, key, k=top_k)
    x_next = x_next[..., None]
    x = jnp.concatenate([x, x_next], axis=1)  # (B, T+1)#

    while x.shape[1] < max_length:
        logits = model(x_next) / temperature
        x_next, key = top_k_sampling(logits, key, k=top_k)
        x = jnp.concatenate((x, x_next), axis=-1)  # (B, T+1)#
    return x


def generate_completions(
    model,
    enc=None,
    prefix="Hello, I'm a language model,",
    num_completions=8,
    max_length=20,
    temperature=0.2,
    key=None,
):
    if enc is None:
        enc = tiktoken.get_encoding("gpt2")
    if key is None:
        key = jax.random.PRNGKey(1337)

    generate_completion = partial(
        generate_slow, model, key=key, max_length=max_length, temperature=temperature
    )
    tokens = enc.encode(prefix)
    tokens = jnp.array(tokens, dtype=jnp.int32)
    tokens = jnp.expand_dims(tokens, axis=0)
    x = jnp.tile(tokens, (num_completions, 1))

    x = generate_completion(x=x)  # Make sure you can do a forward pass
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
