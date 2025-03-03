import jax
import optax
import jax.numpy as jnp
from flax import nnx
import tiktoken

import torch
from transformers import GPT2LMHeadModel

import dataloaders as dl
from models import Charformer
from train import train_step
from infer import top_k_sampling

BLOCK_SIZE = 8


def run_hf_gpt2():
    max_length = 10
    temperature = 0.7
    top_k = 50
    prompt = "Once upon a time,"

    encoder = tiktoken.get_encoding("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # Tokenize input prompt
    input_ids = torch.tensor([encoder.encode(prompt)])

    # PRNG key for randomness in JAX
    key = jax.random.PRNGKey(0)

    generated_tokens = list(input_ids[0].numpy())
    for _ in range(max_length - len(generated_tokens)):
        input_tensor = torch.tensor([generated_tokens])
        outputs = model(input_tensor, attention_mask=torch.ones_like(input_tensor))
        logits = outputs.logits[:, -1, :].detach().numpy()
        logits_jax = jnp.array(logits) / temperature
        next_token, key = top_k_sampling(logits_jax, key, k=top_k)
        generated_tokens.append(int(next_token[0]))

    # Decode the output tokens
    generated_text = encoder.decode(generated_tokens)
    print(generated_text)


def run_charformer():
    text = dl.load_text("datasets/panchatantra-ryder.txt")
    dataloader = dl.CharLoader(text)
    data = dataloader.encode_text(text)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    key = jax.random.PRNGKey(1337)
    rngs = nnx.Rngs({"dataloader": key, "dropout": key, "params": key, "generate": key})

    features = 32
    num_heads = 4
    num_blocks = 3
    m = Charformer(
        dataloader.vocab_size, features, num_heads, num_blocks, BLOCK_SIZE, rngs
    )

    # Generate sample text
    out = m.generate(
        key, jnp.zeros((1, 1), dtype=jnp.int32), BLOCK_SIZE, max_new_tokens=100
    )[0].tolist()
    out = dataloader.decode(out)
    print(out)

    # Train the model
    batch_size = 32
    optimizer = nnx.Optimizer(m, optax.adam(1e-3))

    valid_xb, valid_yb = dataloader.get_batch(key, val_data, len(val_data), BLOCK_SIZE)

    for steps in range(80):
        key = jax.random.split(key)[0]
        xb, yb = dataloader.get_batch(key, train_data, batch_size, BLOCK_SIZE)
        train_loss = train_step(m, optimizer, xb, yb)
        valid_loss = train_step(m, optimizer, valid_xb, valid_yb)
        if steps % 20 == 0:
            print(
                f"step {steps}: train loss {train_loss:.4f} valid loss {valid_loss:.4f}"
            )

    # Generate sample text
    out = m.generate(
        key, jnp.zeros((1, 1), dtype=jnp.int32), BLOCK_SIZE, max_new_tokens=100
    )[0].tolist()
    out = dataloader.decode(out)
    print(out)


if __name__ == "__main__":
    run_hf_gpt2()
