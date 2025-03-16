from typing import Callable
import os

import jax
import numpy as np
import jax.numpy as jnp

import tiktoken


def load_text(path):
    with open(path, "r") as f:
        text = f.read()
    return text


class DataLoader:
    def __init__(
        self,
        dirpath: str,
        batch_size: int,
        block_size: int,
        device_rank: int,
        label: str | None = None,
        quiet: bool = False,
    ):
        self.dirpath = dirpath
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens["<|endoftext|>"]

        self.B = batch_size
        self.T = block_size
        self.D = device_rank

        self.shards = os.listdir(dirpath)
        if label is not None:
            self.shards = [shard for shard in self.shards if label in shard]

        self.cur_shard = 0
        self.shard_pos = 0
        self.shard = self.__load_shard()
        self.shard_size = len(self.shard)

        if not quiet:
            print(f"""dataloader initialized:
------------------------
label:          {label}
shards:         {len(self.shards):,}
shard size:     {self.shard_size:,}
batch size:     {self.B}
block size:     {self.T}
device rank:    {self.D}
------------------------""")

    def __len__(self):
        return len(self.shards) * self.shard_size

    def __load_shard(self):
        if self.cur_shard >= len(self.shards):
            self.cur_shard = 0
        shard = self.shards[self.cur_shard]
        tokens = np.load(os.path.join(self.dirpath, shard))
        if type(tokens) is not np.ndarray:
            tokens = tokens["arr_0"]
        self.shard_size = len(tokens)
        return tokens

    def __call__(self):
        # preallocate the  buffer
        buf_size = self.B * self.T * self.D + 1
        buf = np.zeros((buf_size,), dtype=np.uint16)

        # if the shard has enough tokens remaining
        if self.shard_pos + buf_size < self.shard_size:
            buf[:] = self.shard[self.shard_pos : self.shard_pos + buf_size]
            self.shard_pos += buf_size
        else:
            # load the remaining shard into the buffer
            buf_prefix_size = self.shard_size - self.shard_pos
            buf[:buf_prefix_size] = self.shard[
                self.shard_pos : self.shard_pos + buf_prefix_size
            ]
            buf_pos = buf_prefix_size

            # if the remainder of the buffer is larger than the shard, load as many shards as needed
            if buf_size - buf_prefix_size > self.shard_size:
                n_shards = (buf_size - buf_prefix_size) // self.shard_size
                for _ in range(n_shards):
                    self.cur_shard += 1
                    self.shard = self.__load_shard()
                    buf[buf_pos : buf_pos + self.shard_size] = self.shard
                    buf_pos += self.shard_size

            # Load the next shard
            self.cur_shard += 1
            self.shard = self.__load_shard()
            self.shard_pos = 0

            # load the remainder of the buffer
            buf[buf_pos:] = self.shard[: buf_size - buf_pos]
            self.shard_pos = buf_size - buf_pos

        X = buf[:-1].reshape((self.D, self.B, self.T))
        Y = buf[1:].reshape((self.D, self.B, self.T))

        return jnp.array(X), jnp.array(Y)


class CharLoader:
    def __init__(self, text: str):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: "".join(self.itos[i] for i in l)

    def get_encoder_decoder(self, text) -> tuple[Callable, Callable]:
        return self.encode, self.decode, self.vocab_size

    def encode_text(self, text) -> jax.Array:
        data = jnp.array(self.encode(text), dtype=jnp.int32)
        return data

    def get_batch(self, key, data: jax.Array, batch_size, block_size):
        ix = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
        x = jnp.stack([data[i : i + block_size] for i in ix])
        y = jnp.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y
