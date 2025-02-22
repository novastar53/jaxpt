import jax
import optax
import jax.numpy as jnp
from flax import nnx


import dataloaders as dl
from models import Bigram, Charformer, GPT2, GPTConfig
from train import train_step
from utils import count_params, list_params

BLOCK_SIZE = 8 


def run_gpt2():

    num_return_sequences = 5
    max_length = 50
    
    key = jax.random.PRNGKey(1337)    
    rngs = nnx.Rngs({"dataloader": key, "dropout": key, "params": key, "generate": key})
    m = GPT2.from_pretrained(rngs)

    _, state = nnx.split(m)
    print(count_params(state))

    import tiktoken

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = jnp.array(tokens, dtype=jnp.int32) # (8,)
    tokens = jnp.expand_dims(tokens, axis=0)
    x = jnp.tile(tokens, (num_return_sequences, 1)) # (5, 8)

    while x.shape[1] < max_length: 
        logits = m(x) 
        logits = logits[:, -1, :] # (B, vocab_size)
        probs = jax.nn.softmax(logits, axis=-1)    
        #print(probs.shape)
        topk_probs, topk_indices = jax.lax.top_k(probs, 50)
        #print(topk_probs.shape, topk_indices.shape)
        #print(topk_probs.shape)
        key, _ = jax.random.split(key)
        ix = jax.random.categorical(key, jnp.log(topk_probs), shape=(topk_probs.shape[0],)) # (B, 1)
        ix = ix.reshape(ix.shape[0], 1)
        #print(ix, ix.shape)
        #xcol = jnp.take_along_axis(topk_indices, ix, axis=-1)
        #print(x.shape)
        x = jnp.concatenate((x, ix), axis=1) # (B, T+1)
        #print(x.shape)

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