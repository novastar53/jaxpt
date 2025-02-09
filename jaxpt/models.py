import jax
import jax.numpy as jnp
import flax
import optax
from flax import nnx


class BigramLanguageModel(nnx.Module):

    def __init__(self, vocab_size, rngs: nnx.Rngs):

        self.rngs = rngs
        self.embed = nnx.Embed(num_embeddings=vocab_size, features=vocab_size, rngs=rngs)

    def __call__(self, idx, targets=None):

        logits = self.embed(idx) # (B, T, C)
        return logits


    def generate(self, key, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            key = jax.random.split(key)[0]
            logits = self(idx)
            logits = logits[:, -1, :] # (B, C)
            probs = nnx.softmax(logits, axis=-1) # (B, C)
            idx_next = jax.random.categorical(key, jnp.log(probs), shape=(probs.shape[0], 1)) # (B, 1)
            idx = jnp.concatenate((idx, idx_next), axis=1) # (B, T+1)
        
        return idx
