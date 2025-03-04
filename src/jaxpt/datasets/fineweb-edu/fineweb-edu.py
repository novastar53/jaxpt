import os

import tiktoken
from datasets import load_dataset


local_dir =  "data" 
https://www.youtube.com/watch?v=l8pRSuU81PU
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", cache_dir=DATA_CACHE_DIR)

# tokenize the dataset
tokenizer = tiktoken.get_encoding("gpt2")

