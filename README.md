# JaxPT
GPT models built with JAX

## Getting Started 

This project implements the GPT series of models using Jax and Flax's NNX library.

### Requirements

Install the UV python package managment library

`curl -LsSf https://astral.sh/uv/install.sh | sh`

### Available Commands

The main commands available in the Makefile are:

- `make install` - Install dependencies from lockfile
- `make dev` - Install all dependencies including dev from lockfile
- `make clean` - Clean build artifacts and cache
- `make build` - Build package
- `make lint` - Run linting
- `make format` - Format code
- `make lab` - Run Jupyter lab server from the project directory

To see all available commands and their descriptions, run: `make help`

### Training

The training run can be reproduced using `notebooks/train_gpt2.ipynb`
A machine with 8 x Nvidia A100 80GB GPUs used to train for 1 epoch on a 10bn token sample of the 
Fineweb-Edu dataset. Validation was performed on 1% of the dataset. 
The trained model was evaluated on the Hellaswag benchmark.

### Results

The trained model achieved a score of 0.3025 on the Hellaswag benchmark.

![Training Curve](./gpt2_135mm_fineweb-edu_14_03_2025-1.png)
![Zoomed In](./gpt2_135mm_fineweb-edu_14_03_2025-2.png)




