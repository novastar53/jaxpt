import time
import os
import requests
import json
import pandas as pd

from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import tiktoken
from datasets import load_dataset
import optax

from jaxpt.models import GPT, from_checkpoint

dataset_url =  "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
DATA_CACHE_DIR = Path() / "hellaswag"

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
def _predict(model, tokens, mask):
    logits = model(tokens)
    logits = logits[:, :-1, :] # remove the last token
    logits_flat = logits.reshape(-1, logits.shape[-1])
    tokens = tokens[:, 1:] # remove the first token
    tokens_flat = tokens.reshape(-1)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits_flat, tokens_flat)
    mask = mask[..., 1:]
    losses = losses.reshape(tokens.shape[0], -1)
    losses = losses * mask
    losses = losses.sum(axis=1) / mask.sum(axis=1)
    pred = losses.argmin()
    return pred


class HellaSwag:
    def __init__(self, model: nnx.Module):
        #self.dataset = load_dataset("hellaswag", trust_remote_code=True, num_proc=0)
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
        endings_tokens = [self.tokenizer.encode(ending) for ending in endings]
        mask = jnp.zeros((len(endings_tokens), len(prompt_tokens) + max(len(ending) for ending in endings_tokens)), dtype=jnp.int16)
        tokens = jnp.zeros((len(endings_tokens), len(prompt_tokens) + max(len(ending) for ending in endings_tokens)), dtype=jnp.int16)
        for i in range(len(endings_tokens)):
            tokens = tokens.at[i, :len(prompt_tokens) + len(endings_tokens[i])].set(prompt_tokens + endings_tokens[i])
            mask = mask.at[i, :len(prompt_tokens) + len(endings_tokens[i])].set(jnp.ones(len(prompt_tokens) + len(endings_tokens[i])))

        return tokens, mask, label

    def __len__(self):
        return len(self.valid)



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", default="run_20250313_tywkeh")
    parser.add_argument("--chkpt", default="checkpoint-128.pt")

    args = parser.parse_args()

    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(key)
    checkpoint = Path("/") / "checkpoints" / args.run / args.chkpt
    model = from_checkpoint(checkpoint, rngs=rngs)
    model.eval()

    hellaswag = HellaSwag(model)
    total, correct = 0, 0
    start = time.time()
    for example, pred, label in hellaswag:
        #print(example["ctx"])
        #for ending in example["endings"]:
        #    print(ending)
        #print(pred, label)
        #print("----------")
        print(f"Processed: {total}", end="\r")
        total += 1
        if pred == label:
            correct += 1
        if total % 100 == 0:
            delta = time.time() - start
            rate = total / delta
            print(f"correct {correct} | total {total} | acc {100*correct/total:0.2f}% | rate {rate:0.4f}/sec ")
    
    print(f"Accuracy: {100*correct/total}%")
