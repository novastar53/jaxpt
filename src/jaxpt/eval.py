import time
import os
import sys
import requests
from typing import Callable 
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import tiktoken
import optax

from jaxpt.models import GPT, from_huggingface_pretrained, GPTConfig
from jaxpt.dataloaders import DataLoader
from transformers import FlaxGPT2LMHeadModel, GPT2LMHeadModel

dataset_url = (
    "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
)
DATA_CACHE_DIR = Path() / "hellaswag"


def calc_validation_loss(model: nnx.Module, loss_fn: Callable, dataloader: DataLoader, eval_steps=10):
  valid_loss = 0.0
  for i in range(eval_steps):
    batch, targets = dataloader()
    batch = np.squeeze(batch)
    targets = np.squeeze(targets)
    loss = loss_fn(model, batch, targets)
    valid_loss += loss
  valid_loss /= eval_steps
  return valid_loss


def _download_hellaswag():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_val.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {dataset_url} to {data_filename}...")
        resp = requests.get(dataset_url, stream=True)
        with open(data_filename, "wb") as file:
            for data in resp.iter_content(chunk_size=1024):
                file.write(data)


@nnx.jit
def _predict(model: GPT, tokens, mask):
    logits = model(tokens)
    logits = logits[:, :-1, :]  # remove the last token
    logits_flat = logits.reshape(-1, logits.shape[-1])
    tokens = tokens[:, 1:]  # remove the first token
    tokens_flat = tokens.reshape(-1)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits_flat, tokens_flat)
    mask = mask[..., 1:]
    losses = losses.reshape(tokens.shape[0], -1)
    losses = losses * mask
    losses = losses.sum(axis=1) / mask.sum(axis=1)
    pred = losses.argmin()
    return pred


class HellaSwag:
    def __init__(self, model: GPT2LMHeadModel | FlaxGPT2LMHeadModel | GPT):
        _download_hellaswag()
        self.file = open(os.path.join(DATA_CACHE_DIR, f"hellaswag_val.jsonl"), "r")
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.batch_size = 4
        self.idx = 0
        self.model = model

    def __iter__(self):
        return self

    def __next__(self):
        line = self.file.readline()
        if line:
            example = json.loads(line)
            tokens, mask, label = self.__process_example(example)
            pred = _predict(model, tokens, mask)
            return example, pred, label
        self.file.close()
        raise StopIteration

    def __process_example(self, example):
        prompt = example["ctx"]
        endings = example["endings"]
        label = example["label"]
        prompt_tokens = self.tokenizer.encode(prompt)
        ending_tokens = [self.tokenizer.encode(" " + ending) for ending in endings]

        len_prompt = len(prompt_tokens)
        max_ending_len = max(len(ending) for ending in endings)
        mask = jnp.zeros((len(endings), len_prompt + max_ending_len))
        tokens = jnp.zeros((len(endings), len_prompt + max_ending_len), dtype=jnp.int32)
        for i in range(len(endings)):
            len_tokens = len(prompt_tokens) + len(ending_tokens[i])
            tokens = tokens.at[i, :len_tokens].set(prompt_tokens + ending_tokens[i])
            mask = mask.at[i, :len_tokens].set(jnp.ones(len_tokens))
            mask = mask.at[i, : len(prompt_tokens)].set(jnp.zeros(len(prompt_tokens)))

        return tokens, mask, label

    def __len__(self):
        return len(self.valid)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--run", default="run_20250312_kpsqxx")
    parser.add_argument("--chkpt", default="checkpoint-18882.pt")
    parser.add_argument("--datadir", default="")
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32")
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(key)
    checkpoint = Path().absolute() / "checkpoints" / args.run / args.chkpt
    config = GPTConfig(dtype=jnp.bfloat16, vocab_size=50304, sdpa_implementation="xla")
    model = GPT.from_checkpoint(checkpoint, rngs=rngs, config=config)
    # model = from_huggingface_pretrained(rngs=rngs)
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    hellaswag = HellaSwag(model)

    total, correct = 0, 0
    start = time.time()
    for example, pred, label in hellaswag:
        # print(example["ctx"])
        # for ending in example["endings"]:
        #    print(ending)
        # print(pred, label)
        # print("----------")
        # print(f"Processed: {total}", end="\r")
        total += 1
        correct += int(pred == label)

        if total % 1 == 0:
            delta = time.time() - start
            rate = total / delta
            print(
                f"correct {correct} | total {total} | acc {100 * correct / total:0.2f}% | rate {rate:0.1f}/sec "
            )

    print(f"Accuracy: {100 * correct / total}%")
