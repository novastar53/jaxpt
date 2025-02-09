from typing import Callable

import jax
import jax.numpy as jnp



def load_text(path):
    with open(path, 'r') as f:
        text = f.read()
    return text


def get_encoder_decoder(text) -> tuple[Callable, Callable]:

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [ stoi[c] for c in s ]
    decode = lambda l: ''.join(itos[i] for i in l)

    return encode, decode, vocab_size


def encode_text(text, encode) -> jax.Array:

    data = jnp.array(encode(text), dtype=jnp.int32)
    return data


def get_batch(key, data, batch_size, block_size):

    ix = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y 
