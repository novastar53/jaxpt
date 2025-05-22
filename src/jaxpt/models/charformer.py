import jax
import jax.numpy as jnp
from flax import nnx


class Head(nnx.Module):
    def __init__(self, head_size, features, block_size, rngs: nnx.Rngs):
        self.key = nnx.Linear(features, head_size, rngs=rngs, use_bias=False)
        self.query = nnx.Linear(features, head_size, rngs=rngs, use_bias=False)
        self.value = nnx.Linear(features, head_size, rngs=rngs, use_bias=False)
        # self.tril = jnp.tril(jnp.ones((block_size, block_size)))
        self.dropout = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (batch_size, block_size, head_size)
        q = self.query(x)  # (batch_size, block_size, head_size)

        wei = q @ jnp.transpose(
            k, axes=(0, 2, 1)
        )  # (batch_size, block_size, block_size)
        tril = jnp.tril(jnp.ones((T, T)))
        wei = (
            jnp.where(tril == 0, float("-inf"), wei) * C**-0.5
        )  # (batch_size, block_size, block_size)
        wei = jax.nn.softmax(wei, axis=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # (batch_size, block_size, head_size)
        out = wei @ v  # (batch_size, block_size, head_size)
        return out


class FeedForward(nnx.Module):
    def __init__(self, features, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(features, 4 * features, rngs=rngs)
        self.linear2 = nnx.Linear(4 * features, features, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        out = self.dropout(x)
        return out


class AttentionBlock(nnx.Module):
    def __init__(self, features, num_heads, block_size, rngs: nnx.Rngs):
        self.heads = [
            Head(
                head_size=features // num_heads,
                features=features,
                block_size=block_size,
                rngs=rngs,
            )
            for _ in range(num_heads)
        ]
        self.proj = nnx.Linear(features, features, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
        self.ffwd = FeedForward(features, rngs=rngs)
        self.ln1 = nnx.LayerNorm(features, rngs=rngs)
        self.ln2 = nnx.LayerNorm(features, rngs=rngs)

    def __call__(self, x):
        # self attention
        res = x
        x = self.ln1(x)
        x = jnp.concatenate([h(x) for h in self.heads], axis=-1)
        x = res + self.dropout(self.proj(x))

        # feed forward
        res = x
        x = self.ln2(x)
        out = res + self.ffwd(x)
        return out


class Charformer(nnx.Module):
    def __init__(
        self,
        vocab_size,
        features,
        num_heads,
        num_blocks,
        block_size,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.embed = nnx.Embed(
            num_embeddings=vocab_size, features=features, rngs=rngs
        )
        self.pos_emb = nnx.Embed(block_size, features, rngs=rngs)
        self.blocks = [
            AttentionBlock(features, num_heads, block_size, rngs=rngs)
            for _ in range(num_blocks)
        ]
        self.ln = nnx.LayerNorm(features, rngs=rngs)
        self.lm_head = nnx.Linear(features, vocab_size, rngs=rngs)

    def __call__(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.embed(idx)  # B, T, C
        pos_emb = self.pos_emb(jnp.arange(T))  # T, C
        x = tok_emb + pos_emb  # B, T, C
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.lm_head(x)  # B, T, vocab_size

        return logits

    def generate(self, key, idx, block_size, max_new_tokens):
        for _ in range(max_new_tokens):
            key = jax.random.split(key)[0]
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C)
            probs = nnx.softmax(logits, axis=-1)  # (B, C)
            idx_next = jax.random.categorical(
                key, jnp.log(probs), shape=(probs.shape[0], 1)
            )  # (B, 1)
            idx = jnp.concatenate((idx, idx_next), axis=1)  # (B, T+1)

        return idx
