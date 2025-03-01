from functools import partial

import jax
import jax.numpy as jnp
import flax.nnx as nnx


def top_k_sampling(logits, key, k=50):
    top_k_indices = jnp.argsort(logits, axis=-1)[..., -k:]
    top_k_logits = jnp.take_along_axis(logits, top_k_indices, axis=-1)
    probabilities = jax.nn.softmax(top_k_logits, axis=-1)
    key, subkey = jax.random.split(key)
    sampled_index = jax.random.categorical(subkey, probabilities)
    return jnp.take_along_axis(top_k_indices, sampled_index[..., None], axis=-1).squeeze(-1), key


#@nnx.jit(static_argnames=("max_length", "temperature", "top_k"))
def generate(model: nnx.Module,  *, x: jax.Array, max_length=50,
                        temperature=0.7, top_k = 50) -> jax.Array:

    key = jax.random.PRNGKey(0)

    while x.shape[1] < max_length:
        logits = model(x)[:, -1, :] / temperature
        x_next, key = top_k_sampling(logits, key, k=top_k)
        x_next = x_next.reshape(x_next.shape[0], 1)
        x = jnp.concatenate((x, x_next), axis=1) # (B, T+1)#
    return x

