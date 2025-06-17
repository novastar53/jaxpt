from typing import Callable, List
import os
from abc import ABC, abstractmethod

import jax
import numpy as np
import jax.numpy as jnp

import tiktoken


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


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders that yield token batches.
    Subclasses must implement _list_shards and _load_shard.
    """

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        device_rank: int,
        label: str | None = None,
        quiet: bool = False,
    ):
        # Common initialization
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens["<|endoftext|>"]
        self.B = batch_size
        self.T = block_size
        self.D = device_rank
        self.label = label

        # List and filter shards
        self.shards = self._list_shards(label)
        self.cur_shard = 0
        self.shard_pos = 0
        self.shard = self._load_shard()
        self.shard_size = len(self.shard)

        if not quiet:
            print(f"""{self.__class__.__name__} initialized:
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

    def __call__(self):
        # preallocate buffer
        buf_size = self.B * self.T * self.D + 1
        buf = np.zeros((buf_size,), dtype=np.uint16)

        if self.shard_pos + buf_size < self.shard_size:
            buf[:] = self.shard[self.shard_pos : self.shard_pos + buf_size]
            self.shard_pos += buf_size
        else:
            # fill the remaining shard
            buf_prefix = self.shard_size - self.shard_pos
            buf[:buf_prefix] = self.shard[self.shard_pos :]
            buf_pos = buf_prefix

            # load the next shard
            self.cur_shard += 1
            self.shard = self._load_shard()
            self.shard_pos = 0

            # fill full shards
            while buf_pos + self.shard_size <= buf_size:
                buf[buf_pos : buf_pos + self.shard_size] = self.shard
                buf_pos += self.shard_size
                self.cur_shard += 1
                self.shard = self._load_shard()

            # final partial shard
            self.shard_pos = buf_size - buf_pos
            buf[buf_pos:] = self.shard[: self.shard_pos]

        X = buf[:-1].reshape((self.D, self.B, self.T))
        Y = buf[1:].reshape((self.D, self.B, self.T))
        return jnp.array(X), jnp.array(Y)

    @abstractmethod
    def _list_shards(self, label: str | None) -> list[str]:
        """Return list of shard identifiers, filtered by label."""
        pass

    @abstractmethod
    def _load_shard(self) -> np.ndarray:
        """Load and return current shard as a 1D numpy array."""
        pass


class DataLoader(BaseDataLoader):
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
        super().__init__(batch_size, block_size, device_rank, label, quiet)

    def _list_shards(self, label):
        shards = os.listdir(self.dirpath)
        if label is not None:
            shards = [s for s in shards if label in s]
        return shards

    def _load_shard(self):
        if self.cur_shard >= len(self.shards):
            self.cur_shard = 0
        shard = self.shards[self.cur_shard]
        tokens = np.load(os.path.join(self.dirpath, shard))
        if not isinstance(tokens, np.ndarray):
            tokens = tokens["arr_0"]
        self.shard_size = len(tokens)
        return tokens


class CloudDataLoader(BaseDataLoader):
    """
    DataLoader that reads token shards from a Google Cloud Storage bucket.
    """

    def __init__(
        self,
        bucket_name: str,
        bucket_prefix: str,
        batch_size: int,
        block_size: int,
        device_rank: int,
        label: str | None = None,
        quiet: bool = False,
    ):
        from google.cloud import storage

        self.bucket_name = bucket_name
        self.bucket_prefix = bucket_prefix
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        super().__init__(batch_size, block_size, device_rank, label, quiet)

    def _list_shards(self, label):
        blobs = self.bucket.list_blobs(prefix=self.bucket_prefix)
        return [
            blob.name for blob in blobs if (label is None or label in blob.name)
        ]

    def _load_shard(self):
        from io import BytesIO

        if self.cur_shard >= len(self.shards):
            self.cur_shard = 0
        shard_name = self.shards[self.cur_shard]
        blob = self.bucket.blob(shard_name)
        data = blob.download_as_bytes()
        tokens = np.load(BytesIO(data))
        if not isinstance(tokens, np.ndarray):
            tokens = tokens["arr_0"]
        self.shard_size = len(tokens)
        return tokens


class HuggingfaceDataLoader(BaseDataLoader):
    def __init__(
        self,
        batch_size: int,
        block_size: int,
        device_rank: int,
        tokenizer: str,
        dataset_paths: List[str],
        dataset_names: List[str],
        probabilities: List[float],
        label: str,
        random_seed: int = 42,
        buffer_size: int = 10_000,
        streaming: bool = True,
        quiet: bool = False,
    ):
        self.streaming = True

        from datasets import load_dataset, interleave_datasets
        from transformers import AutoTokenizer

        print(f"Initializing tokenizer {tokenizer}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.dataset_objs = []
        for ds_path, ds_name in zip(dataset_paths, dataset_names, strict=False):
            self.dataset_objs.append(
                load_dataset(ds_path, ds_name, split=label, streaming=streaming)
            )
        ds = interleave_datasets(
            self.dataset_objs, probabilities=probabilities, seed=random_seed
        )
        self.ds = ds.shuffle(buffer_size=buffer_size, seed=random_seed)
        self.iter_ds = iter(self.ds)

        super().__init__(batch_size, block_size, device_rank, label, quiet)

    def _list_shards(self, label):
        if self.streaming is True:
            return []
        return NotImplementedError

    def _load_shard(self):
        try:
            example = next(self.iter_ds)
        except StopIteration:
            self.iter_ds = iter(self.ds)
            example = next(self.iter_ds)

        while example["text"] is None:
            try:
                example = next(self.iter_ds)
            except StopIteration:
                self.iter_ds = iter(self.ds)
                example = next(self.iter_ds)

        tokens = self.tokenizer.encode(example["text"])
        self.shard_size = len(tokens)
        return jnp.array(tokens, dtype=jnp.uint16)


class SFT_CloudDataLoader:
    def __init__(
        self,
        bucket_name: str,
        bucket_prefix: str,
        batch_size: int,
        block_size: int,
        device_rank: int,
        label: str | None = None,
        quiet: bool = False,
    ):
        from google.cloud import storage

        self.bucket_name = bucket_name
        self.bucket_prefix = bucket_prefix
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.batch_size = batch_size
        self.block_size = block_size
        self.device_rank = device_rank
        self.label = label
        self.quiet = quiet

        self.cur_shard = 0
        self.shard_pos = 0
        self.shards = self._list_shards(label)
        self.shard = self._load_shard()
        self.shard_size = len(self.shard)

    def __call__(self):
        buffer = np.empty(
            (self.device_rank * self.batch_size, 3, 1 + self.block_size),
            dtype=np.uint16,
        )
        if (
            self.shard_pos + self.device_rank * self.batch_size
            <= self.shard_size
        ):
            buffer[:] = self.shard[
                self.shard_pos : self.shard_pos
                + self.device_rank * self.batch_size,
                :,
                :,
            ]
            self.shard_pos += self.device_rank * self.batch_size
        else:
            # use up the remaining shard
            rem = self.shard_size - self.shard_pos
            buffer[:rem] = self.shard[self.shard_pos :, :, :]
            # load the next shard
            self.shard = self._load_shard()
            self.cur_shard += 1
            self.shard_pos = 0
            # fill the remaining buffer
            buffer[rem:] = self.shard[
                : self.device_rank * self.batch_size - rem, :, :
            ]
            self.shard_pos = self.device_rank * self.batch_size - rem
        x = buffer[:, 0, :-1].reshape(
            (self.device_rank, self.batch_size, self.block_size)
        )
        y = buffer[:, 0, 1:].reshape(
            (self.device_rank, self.batch_size, self.block_size)
        )
        attn_mask = (
            buffer[:, 1, :-1]
            .reshape((self.device_rank, self.batch_size, self.block_size))
            .astype(np.bool_)
        )
        loss_mask = (
            buffer[:, 2, 1:]
            .reshape((self.device_rank, self.batch_size, self.block_size))
            .astype(np.bool_)
        )
        return (
            jnp.array(x),
            jnp.array(y),
            jnp.array(attn_mask),
            jnp.array(loss_mask),
        )

    def _list_shards(self, label):
        blobs = self.bucket.list_blobs(prefix=self.bucket_prefix)
        return [
            blob.name for blob in blobs if (label is None or label in blob.name)
        ]

    def _load_shard(self):
        from io import BytesIO

        if self.cur_shard >= len(self.shards):
            self.cur_shard = 0
        shard_name = self.shards[self.cur_shard]
        blob = self.bucket.blob(shard_name)
        data = blob.download_as_bytes()
        tokens = np.load(BytesIO(data))
        if not isinstance(tokens, np.ndarray):
            tokens = tokens["arr_0"]
        self.shard_size = len(tokens)
        return tokens
