import jax
import optax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import tiktoken

import torch
from transformers import GPT2LMHeadModel

import dataloaders as dl
from models import Bigram, Charformer, GPT2, GPTConfig, KGPT
from train import train_step
from utils import count_params, list_params, get_param

BLOCK_SIZE = 8 

def top_k_sampling(logits, key, k=50):
    """
    Apply top-k sampling to the logits distribution.
    """
    top_k_indices = jnp.argsort(logits, axis=-1)[..., -k:]
    top_k_logits = jnp.take_along_axis(logits, top_k_indices, axis=-1)
    probabilities = jax.nn.softmax(top_k_logits, axis=-1)
    key, subkey = jax.random.split(key)
    sampled_index = jax.random.categorical(subkey, probabilities)
    return jnp.take_along_axis(top_k_indices, sampled_index[..., None], axis=-1).squeeze(-1), key


def compare_gpts():

    kgpt = KGPT.from_pretrained("gpt2")
    kgpt.eval()
    kgpt_params = kgpt.state_dict()
    

    key = jax.random.PRNGKey(0)    
    rngs = nnx.Rngs({"dataloader": key, "dropout": key, "params": key, "generate": key})
    m, _ = GPT2.from_pretrained(rngs)
    m.eval()
    graphdef, state = nnx.split(m)
    m_params = list_params(state)

    print(len(kgpt_params), len(m_params))
    kgpt_param = kgpt_params['lm_head.weight']
    param = get_param(state, 'lm_head.kernel').value
    print(kgpt_param.shape, param.shape)
    print(torch.sum(kgpt_param), jnp.sum(param))

    kgpt_x = kgpt(torch.tensor([[0,0,0]]))
    print(kgpt_x.shape)
    print(kgpt_x)

    x = m(jnp.array([[0,0,0]]))
    print(x.shape)
    print(x)


def run_karpathy_gpt2():

    max_length = 20
    num_return_sequences = 5
    temperature = 0.7
    top_k = 50
    prompt = "Once upon a time,"

    key = jax.random.PRNGKey(0)
    model = KGPT.from_pretrained("gpt2")
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor([enc.encode(prompt)])
    tokens = jnp.array(tokens.numpy())
    x = jnp.tile(tokens, (num_return_sequences, 1))

    while x.shape[1] < max_length:
        logits, _ = model(torch.tensor(x))
        logits = logits.detach().numpy()[:, -1, :]
        logits_jax = jnp.array(logits) / temperature
        x_next, key = top_k_sampling(logits_jax, key, k=top_k)
        x_next = x_next.reshape(x_next.shape[0], 1)
        x = jnp.concatenate((x, x_next), axis=1)
    
    # Decode the output tokens
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)



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
    attention_mask = torch.ones_like(input_ids)
    
    # Convert input_ids to JAX array
    input_ids_jax = jnp.array(input_ids.numpy())
    
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


def run_gpt2():

    max_length = 20
    num_return_sequences = 5
    temperature = 0.7
    top_k = 50
    prompt = "Once upon a time,"
    
    key = jax.random.PRNGKey(0)    
    rngs = nnx.Rngs({"dataloader": key, "dropout": key, "params": key, "generate": key})
    m, _ = GPT2.from_pretrained(rngs)
    m.eval()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = jnp.array(tokens, dtype=jnp.int32) 
    tokens = jnp.expand_dims(tokens, axis=0)
    x = jnp.tile(tokens, (num_return_sequences, 1)) 

    while x.shape[1] < max_length: 
        logits = m(x) 
        logits = logits[:, -1, :] / temperature # (B, vocab_size)
        x_next, key = top_k_sampling(logits, key, k=top_k)
        x_next = x_next.reshape(x_next.shape[0], 1)
        x = jnp.concatenate((x, x_next), axis=1) # (B, T+1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)



def run_charformer():

    text = dl.load_text("datasets/panchatantra-ryder.txt")
    encode, decode, vocab_size = dl.get_encoder_decoder(text)
    data = dl.encode_text(text, encode)

    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    key = jax.random.PRNGKey(1337)    
    rngs = nnx.Rngs({"dataloader": key, "dropout": key, "params": key, "generate": key})

    features = 32
    num_heads = 4
    num_blocks = 3
    m = Charformer(vocab_size, features, num_heads, num_blocks, BLOCK_SIZE, rngs)

    # Generate sample text
    out = m.generate(key, jnp.zeros((1, 1), dtype=jnp.int32), BLOCK_SIZE, max_new_tokens=100)[0].tolist()
    out = decode(out)
    print(out)

    # Train the model
    batch_size = 32
    optimizer = nnx.Optimizer(m, optax.adam(1e-3))

    valid_xb, valid_yb = dl.get_batch(key, val_data, len(val_data), BLOCK_SIZE)

    for steps in range(5000):
        key = jax.random.split(key)[0]
        xb, yb = dl.get_batch(key, train_data, batch_size, BLOCK_SIZE)
        train_loss = train_step(m, optimizer, xb, yb)
        valid_loss = train_step(m, optimizer, valid_xb, valid_yb)
        if steps % 20 == 0:
            print(f"step {steps}: train loss {train_loss:.4f} valid loss {valid_loss:.4f}")

    # Generate sample text 
    out = m.generate(key, jnp.zeros((1, 1), dtype=jnp.int32), BLOCK_SIZE, max_new_tokens=100)[0].tolist()
    out = decode(out)
    print(out)

if __name__ == "__main__":
    run_gpt2()