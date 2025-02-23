from typing import Callable

import jax
import jax.numpy as jnp

import tiktoken


def load_text(path):
    with open(path, 'r') as f:
        text = f.read()
    return text

class GPTLoader:
    def __init__(self, text: str):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

    def encode_text(self, text) -> jax.Array:
        data = jnp.array(self.encode(text), dtype=jnp.int32)
        return data

    def get_batch(self, key, data: jax.Array, batch_size, block_size):
        ix = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
        x = jnp.stack([data[i:i+block_size] for i in ix])
        y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y 


class CharLoader:

    def __init__(self, text: str):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.encode = lambda s: [ self.stoi[c] for c in s ]
        self.decode = lambda l: ''.join(self.itos[i] for i in l)

    def get_encoder_decoder(self, text) -> tuple[Callable, Callable]:
        return self.encode, self.decode, self.vocab_size

    def encode_text(self, text) -> jax.Array:
        data = jnp.array(self.encode(text), dtype=jnp.int32)
        return data

    def get_batch(self, key, data: jax.Array, batch_size, block_size):
        ix = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
        x = jnp.stack([data[i:i+block_size] for i in ix])
        y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y 
