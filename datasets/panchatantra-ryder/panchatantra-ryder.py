import os
from pathlib import Path

import tiktoken

import numpy as np


if __name__ == "__main__":
    fpath = os.path.join(Path(), "data", "panchatantra-ryder.txt")
    with open(fpath, "r") as f:
        txt = f.read()
        enc = tiktoken.get_encoding("gpt2")
        eot = enc._special_tokens['<|endoftext|>']
        tokens = [eot]
        tokens += enc.encode_ordinary(txt)
        tokens = np.array(tokens)
        train_tokens = tokens[:int(0.9*len(tokens))]
        valid_tokens = tokens[int(0.9*len(tokens)):]
        output_path = os.path.join(Path(), "processed", "panchatantra_train_0.npy")
        np.save(output_path, train_tokens)
        output_path = os.path.join(Path(), "processed", "panchatantra_valid_1.npy")
        np.save(output_path, valid_tokens)
