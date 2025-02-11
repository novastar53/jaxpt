import jax
import optax
import jax.numpy as jnp
from flax import nnx


import dataloaders as dl
from models import Bigram, BasicTransformer
from train import train_step

BATCH_SIZE = 4 # How many independent sequences will we process in parallel?
BLOCK_SIZE = 8 # What is the maximum context length for predictions?

def main():

    text = dl.load_text("datasets/panchatantra-ryder.txt")
    encode, decode, vocab_size = dl.get_encoder_decoder(text)
    data = dl.encode_text(text, encode)

    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    key = jax.random.PRNGKey(1337)    
    rngs = nnx.Rngs({"dataloader": key, "params": key, "generate": key})

    xb, yb = dl.get_batch(key, train_data, BATCH_SIZE, BLOCK_SIZE)

    features = 32
    head_size = 32
    m =  BasicTransformer(vocab_size, features, head_size, BLOCK_SIZE, rngs)

    # Generate sample text
    out = m.generate(key, jnp.zeros((1, 1), dtype=jnp.int32), BLOCK_SIZE, max_new_tokens=100)[0].tolist()
    out = decode(out)
    print(out)

    #return

    # Train the model
    
    batch_size = 32
    optimizer = nnx.Optimizer(m, optax.adam(1e-2))

    for steps in range(100):
        key = jax.random.split(key)[0]
        xb, yb = dl.get_batch(key, train_data, batch_size, BLOCK_SIZE)
        loss = train_step(m, optimizer, xb, yb)
        print(loss)

    # Generate sample text 
    out = m.generate(key, jnp.zeros((1, 1), dtype=jnp.int32), BLOCK_SIZE, max_new_tokens=100)[0].tolist()
    out = decode(out)
    print(out)

if __name__ == "__main__":
    main()