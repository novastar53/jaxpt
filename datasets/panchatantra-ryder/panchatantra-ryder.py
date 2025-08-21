import os
from pathlib import Path

import tiktoken

import numpy as np


if __name__ == "__main__":
    cwd = Path("datasets/panchatantra-ryder").absolute()
    fpath = os.path.join(cwd, "data", "panchatantra-ryder-clean.txt")
    with open(fpath, "r") as f:
        txt = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode_ordinary(txt)
        eot = enc._special_tokens['<|endoftext|>']
        tokens += [eot]
        tokens = np.array(tokens)
        train_tokens = tokens[:int(0.9*len(tokens))]
        valid_tokens = tokens[int(0.9*len(tokens)):]
        output_path = os.path.join(cwd, "processed", "panchatantra_train_0.npy")
        np.save(output_path, train_tokens)
        output_path = os.path.join(cwd, "processed", "panchatantra_valid_1.npy")
        np.save(output_path, valid_tokens)
