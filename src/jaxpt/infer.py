import sys
from functools import partial

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import tiktoken


def top_k_sampling(logits, key, k=50):
    top_k_logits, top_k_indices = jax.lax.top_k(logits, k=k) # Get the top k logits and indices
    key, subkey = jax.random.split(key)  # generate a new random key
    top_k_logit_idxs = jax.random.categorical(subkey, top_k_logits) # sample from the top k logits
    top_k_logit_idxs = top_k_logit_idxs[..., None] # expand dims
    sample_idxs = jnp.take_along_axis(top_k_indices, top_k_logit_idxs, axis=-1) # pick out the sampled tokens
    sample_idxs = sample_idxs[..., 0] # squeeze dims
    return sample_idxs, key


def generate_slow(
    model: nnx.Module, *, x: jax.Array, 
    key: jax.random.PRNGKey,
    max_length=50, 
    temperature=0.2, 
    top_k=50
) -> jax.Array:

    while x.shape[1] < max_length:
        logits = model(x) / temperature
        logits = logits[:, -1, :]
        x_next, key = top_k_sampling(logits, key, k=top_k)
        x_next = x_next[..., None]
        x = jnp.concatenate((x, x_next), axis=1)  # (B, T+1)#
    return x

def generate(
    model: nnx.Module, *, x: jax.Array, 
    key: jax.random.PRNGKey,
    max_length=50, 
    temperature=0.2, 
    top_k=50
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


def generate_completions(model, 
                         enc=tiktoken.get_encoding("gpt2"),
                         prefix="Hello, I'm a language model,", 
                         num_completions=5, max_length=20, 
                         key=jax.random.PRNGKey(1337)):

    generate_completion = partial(generate_slow, model, key=key, max_length=max_length)
    tokens = enc.encode(prefix)
    tokens = jnp.array(tokens, dtype=jnp.int32)
    tokens = jnp.expand_dims(tokens, axis=0)
    x = jnp.tile(tokens, (num_completions, 1))

    x = generate_completion(x=x) # Make sure you can do a forward pass
    output = []
    for i in range(num_completions):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        output.append(decoded)
    return output


def generate_chat(model, 
                    enc=tiktoken.get_encoding("gpt2"),
                    prefix="Hello, I'm a language model,", 
                    temperature=0.2,
                    top_k=50,
                    key=jax.random.PRNGKey(1337)):


    x = enc.encode(prefix)
    x = jnp.array(x, dtype=jnp.int32)
    x = x[None, ...]

    print("Model: ", end="")
    try:
        while True:
            logits = model(x)[:, -1, :] / temperature
            key, subkey = jax.random.split(key)
            x_next, key = top_k_sampling(logits, subkey, k=top_k)
            if x_next[0] == enc.eos_token_id:
                break
            decoded = enc.decode(x_next)
            print(decoded, end="")
            sys.stdout.flush()
            x_next = x_next[..., None]
            x = jnp.concatenate((x, x_next), axis=1)  # (B, T+1)#
    except KeyboardInterrupt:
        print("\n")
