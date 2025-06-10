import time 
import os
from functools import partial

import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import jax.numpy as jnp

import flax.nnx as nnx
import optax

import tensorflow_datasets as tfds

from jaxpt.utils import count_params

#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

print(f"num devices: {jax.device_count()}")

VOCAB_SIZE = 50000
BATCH_SIZE = 32
BLOCK_SIZE = 1024

NUM_LAYERS = 16

EMBED_DIM = 1024
FF_DIM = EMBED_DIM * 4
NUM_HEADS = 8
HEAD_DIM = EMBED_DIM // NUM_HEADS


DATA_DIMS = 2
MODEL_DIMS = 4

LEARNING_RATE = 1e-4

DTYPE = jnp.bfloat16

key = jax.random.key(1337)

lecun_init = nnx.with_partitioning(
    nnx.initializers.lecun_normal(dtype=DTYPE),
    (None, "model")
)
zeros_init = nnx.with_partitioning(
    nnx.initializers.zeros_init(),
    ("model",)
)


class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, dtype=jnp.float32):

        self.fc = nnx.Linear(
            in_features=EMBED_DIM,
            out_features=FF_DIM,
            param_dtype=dtype,
            kernel_init=lecun_init,
            bias_init=zeros_init,
            rngs=rngs,
        ) 

        self.proj = nnx.Linear(
            in_features=FF_DIM,
            out_features=EMBED_DIM,
            param_dtype=dtype,
            kernel_init=lecun_init,
            bias_init=zeros_init,
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
            kernel_init=lecun_init,
            bias_init=zeros_init,
            param_dtype=dtype,
            rngs=rngs
        )
        self.k_proj = nnx.Linear(
            in_features=EMBED_DIM,
            out_features=EMBED_DIM,
            kernel_init=lecun_init,
            bias_init=zeros_init,
            param_dtype=dtype,
            rngs=rngs
        )
        self.v_proj = nnx.Linear(
            in_features=EMBED_DIM,
            out_features=EMBED_DIM,
            kernel_init=lecun_init,
            bias_init=zeros_init,
            param_dtype=dtype,
            rngs=rngs
        )
        self.out_proj = nnx.Linear(
            in_features=EMBED_DIM,
            out_features=EMBED_DIM,
            kernel_init=lecun_init,
            bias_init=zeros_init,
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
                             embedding_init=lecun_init,
                             rngs=rngs)

        self.embed = nnx.Embed(VOCAB_SIZE, 
                                EMBED_DIM, 
                                param_dtype=dtype,
                                embedding_init=lecun_init,
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


@nnx.jit
def create_sharded_model():
    rngs = nnx.Rngs(0)
    model = Model(rngs)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model


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


if __name__ == "__main__":
    mesh = jax.sharding.Mesh(np.reshape(jax.devices(), (DATA_DIMS, MODEL_DIMS)),
                              ["data", "model"])
    
    data_sharding = NamedSharding(mesh, PartitionSpec("data", None))

    with mesh:
        sharded_model = create_sharded_model()
        total_params = count_params(sharded_model)
        print(f"Parameter Count: {total_params:,}")

        
        ds = tfds.load('lm1b', split='train', shuffle_files=False)
        ds = ds.batch(BATCH_SIZE)
        tx = optax.adam(learning_rate=LEARNING_RATE)
        optimizer = nnx.Optimizer(sharded_model, tx)
        iter = 0

        for example in ds:
            last_step_time = time.time()
            ascii_text = convert_to_ascii(example['text'].numpy(), BLOCK_SIZE)
            inputs = ascii_text[:, :-1]
            labels = ascii_text[:, 1:]
            inputs = jax.device_put(inputs, data_sharding)
            labels = jax.device_put(inputs, data_sharding)
            loss = step_fn(sharded_model, optimizer, inputs, labels)
            if iter % 10 == 0:
                new_time = time.time()
                time_elapsed_seconds = (new_time - last_step_time)
                print(f"{iter}, {loss}, {time_elapsed_seconds}")
            iter += 1