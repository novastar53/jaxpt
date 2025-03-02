from typing import Callable

import jax
import jax.numpy as jnp

import tiktoken


def load_text(path):
    with open(path, 'r') as f:
        text = f.read()
    return text


class DataLoader:

    def __init__(self, fpath: str, batch_size: int, block_size: int):
        text = load_text(fpath)
        self.tokens = jnp.array(tiktoken.get_encoding("gpt2").encode(text))
        self.B = batch_size
        self.T = block_size
        self.n = len(self.tokens)
        self.pos = 0

        print(f"dataLoader initialized:")
        print("------------------------")
        print(f"tokens:         {self.n}")
        print(f"batch size:     {self.B}")
        print(f"block size:     {self.T}")
        print("------------------------")

    def __call__(self):
        buf = self.tokens[self.pos:self.pos+self.B*self.T+1]
        if len(buf) < self.B*self.T+1:
            buf = jnp.pad(buf, (0, self.B*self.T+1 - len(buf)), mode='constant', constant_values=0)
        X = buf[:-1].reshape((self.B, self.T))
        Y = buf[1:].reshape((self.B, self.T))
        self.pos += self.B*self.T
        if self.pos >= self.n:
            self.pos = 0
        return X, Y, self.pos

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
