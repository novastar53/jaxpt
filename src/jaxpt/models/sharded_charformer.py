import numpy as np
import jax
import jax.numpy as jnp

import flax.nnx as nnx
import optax

import tensorflow_datasets as tfds

VOCAB_SIZE = 256
BATCH_SIZE = 384
BLOCK_SIZE = 128

EMBED_DIM = 512
FF_DIM = 2048
HEAD_DIM = 128

NUM_LAYERS = 4
NUM_HEADS = 4

LEARNING_RATE = 1e-3


def loss_fn(model, inputs, labels):
    logits = model(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits,
                                                           labels=labels)
    return loss.mean()
    

@nnx.jit
def step_fn(model, optimizer, inputs, labels):
    loss, grads = nnx.value_and_grad(loss_fn)(model, inputs, labels)
    optimizer.update(grads)
    return loss


class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, dtype=jnp.float32):

        self.fc = nnx.Linear(
            in_features=EMBED_DIM,
            out_features=FF_DIM,
            param_dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(dtype=dtype),
            rngs=rngs
        ) 

        self.proj = nnx.Linear(
            in_features=FF_DIM,
            out_features=EMBED_DIM,
            param_dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(dtype=dtype),
            rngs=rngs
        )
    
    def __call__(self, x):
        x = self.fc(x)
        x = nnx.relu(x)
        x = self.proj(x)
        return x


class Attention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, dtype=jnp.float32):
        self.q_proj = nnx.Linear(
            in_features=EMBED_DIM,
            out_features=EMBED_DIM,
            kernel_init=nnx.initializers.lecun_normal(dtype=dtype),
            param_dtype=dtype,
            rngs=rngs
        )
        self.k_proj = nnx.Linear(
            in_features=EMBED_DIM,
            out_features=EMBED_DIM,
            kernel_init=nnx.initializers.lecun_normal(dtype=dtype),
            param_dtype=dtype,
            rngs=rngs
        )
        self.v_proj = nnx.Linear(
            in_features=EMBED_DIM,
            out_features=EMBED_DIM,
            kernel_init=nnx.initializers.lecun_normal(dtype=dtype),
            param_dtype=dtype,
            rngs=rngs
        )
        self.out_proj = nnx.Linear(
            in_features=EMBED_DIM,
            out_features=EMBED_DIM,
            kernel_init=nnx.initializers.lecun_normal(dtype=dtype),
            param_dtype=dtype,
            rngs=rngs
        )
    
    def __call__(self, x):
        
        q = self.q_proj(x).reshape(BATCH_SIZE, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
        k = self.k_proj(x).reshape(BATCH_SIZE, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
        v = self.v_proj(x).reshape(BATCH_SIZE, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)

        _weights_unnormalized = jnp.einsum("BSHD,BTHD->BHST", q, k)
        _weights = jax.nn.softmax(_weights_unnormalized)
        output = jnp.einsum("BHST,BTHD->BSHD", _weights, v)
        output = output.reshape(BATCH_SIZE, BLOCK_SIZE, EMBED_DIM)
        output = self.out_proj(output)
        return output

class Block(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, dtype=jnp.float32):
        self.attn = Attention(rngs, dtype)
        self.mlp = MLP(rngs, dtype)
    
    def __call__(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Model(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, dtype=jnp.float32):
        self.pos = nnx.Embed(BLOCK_SIZE,
                             EMBED_DIM,
                             param_dtype=dtype,
                             embedding_init=nnx.initializers.lecun_normal(dtype=dtype),
                             rngs=rngs)

        self.embed = nnx.Embed(VOCAB_SIZE, 
                                EMBED_DIM, 
                                param_dtype=dtype,
                                embedding_init=nnx.initializers.lecun_normal(dtype=dtype),
                                rngs=rngs)
        self.layers = [ 
            Block(rngs, dtype)
            for _ in range(NUM_LAYERS)
        ]
    
    def __call__(self, x):
        _, T = x.shape
        pos = jnp.arange(0, T, dtype=jnp.uint16)
        pos_x = self.pos(pos)
        x = self.embed(x)
        x = x + pos_x
        for layer in self.layers:
            x = layer(x)
        x = self.embed.attend(x)
        return x


def convert_to_ascii(lines, block_size):
    result = np.zeros(((len(lines), block_size+1)), dtype=np.uint8)
    for line_idx, line in enumerate(lines):
        for chr_idx, chr in enumerate(line):
            if chr_idx >= block_size+1:
                break
            result[line_idx, chr_idx] = chr
    
    return result


if __name__ == "__main__":
    ds = tfds.load('lm1b', split='train', shuffle_files=False)
    ds = ds.batch(BATCH_SIZE)

    rngs = nnx.Rngs(0)
    model = Model(rngs, jnp.float32)
    tx = optax.adam(learning_rate=LEARNING_RATE)
    optimizer = nnx.Optimizer(model, tx)
    iter = 0
    for example in ds:
        ascii_text = convert_to_ascii(example['text'].numpy(), BLOCK_SIZE)
        inputs = ascii_text[:, :-1]
        labels = ascii_text[:, 1:]
        loss = step_fn(model, optimizer, inputs, labels)
        print(f"{iter}, {loss}")
        iter += 1
    
