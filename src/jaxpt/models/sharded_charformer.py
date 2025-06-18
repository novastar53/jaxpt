import time 
import os
from functools import partial
import shutil
from pathlib import Path

import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu import flash_attention as pallas_attention

import flax.nnx as nnx
import optax

import orbax.checkpoint as ocp

import tensorflow_datasets as tfds

from jaxpt.utils import count_params

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

print(f"num devices: {jax.device_count()}")

VOCAB_SIZE = 500 # 50000
BATCH_SIZE = 8
BLOCK_SIZE = 2048

NUM_LAYERS = 2 # 160

EMBED_DIM = 1024
FF_DIM = EMBED_DIM * 4
NUM_HEADS = 8
HEAD_DIM = EMBED_DIM // NUM_HEADS

DATA_DIMS = 2
MODEL_DIMS = 4

LEARNING_RATE = 1e-4

DTYPE = jnp.bfloat16

key = jax.random.key(1337)
rngs = nnx.Rngs(key)

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


def attention_with_masking(_Q, _K, _V, seq_pos=BLOCK_SIZE):
    _, KV_SIZE, _, _ = _K.shape
    query_segment_id = jnp.ones( (1,_Q.shape[1]), dtype=jnp.int32)
    kv_segment_id = jnp.ones( (1, KV_SIZE), jnp.int32) * jnp.expand_dims(jnp.arange(KV_SIZE) <= seq_pos, axis = 0)

    segment_ids = pallas_attention.SegmentIds( 
        q = query_segment_id, 
        kv = kv_segment_id
    )
    return jax.numpy.swapaxes(
        pallas_attention.mha_reference(
            jax.numpy.swapaxes(_Q,1,2), 
            jax.numpy.swapaxes(_K,1,2), 
            jax.numpy.swapaxes(_V,1,2), 
            None, 
            segment_ids = segment_ids),
            1,2
        )


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
        self.key_cache = nnx.Variable(jnp.zeros((1, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)))
        self.value_cache = nnx.Variable(jnp.zeros((1, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)))
    
    def __call__(self, x, pos=0, use_cache=False):
        B, T, _ = x.shape
        
        q = self.q_proj(x).reshape(B, T, NUM_HEADS, HEAD_DIM)
        k = self.k_proj(x).reshape(B, T, NUM_HEADS, HEAD_DIM)
        v = self.v_proj(x).reshape(B, T, NUM_HEADS, HEAD_DIM)

        #print(pos, jnp.sum(self.key_cache.value))

        if use_cache:
            self.key_cache.value = jax.lax.dynamic_update_index_in_dim(
                self.key_cache.value, k, pos, 1)
            self.value_cache.value = jax.lax.dynamic_update_index_in_dim(
                self.value_cache.value, v, pos, 1)
            _weights_unnormalized = jnp.einsum(
                "BSHD,BTHD->BHST", q, self.key_cache)
            _weights = jax.nn.softmax(_weights_unnormalized)
            #output = jnp.einsum("BHST,BTHD->BSHD", _weights, self.value_cache)
            output = attention_with_masking(q, self.key_cache, self.value_cache, 
                                            seq_pos=pos+T)
        else:
            _weights_unnormalized = jnp.einsum("BSHD,BTHD->BHST", q, k)
            _weights = jax.nn.softmax(_weights_unnormalized)
            #output = jnp.einsum("BHST,BTHD->BSHD", _weights, v)
            output = attention_with_masking(q, k, v, seq_pos=pos+T)

        output = output.reshape(B, T, EMBED_DIM)
        output = self.out_proj(output)
        return output


class Block(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, dtype=jnp.float32):
        self.attn = Attention(rngs, dtype)
        self.mlp = MLP(rngs, dtype)
    
    def __call__(self, x, pos=0, use_cache=False):
        x = x + self.attn(x, pos=pos, use_cache=use_cache)
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
    
    def __call__(self, x, pos=0, use_cache=False):
        _, T = x.shape
        idx = jnp.arange(pos, pos+T, dtype=jnp.uint16)
        pos_x = self.pos(idx)
        x = self.embed(x)
        x = x + pos_x
        for layer in self.layers:
            x = layer(x, pos=pos, use_cache=use_cache)
        x = self.embed.attend(x)
        return x


def prepare_train_batch(lines, block_size):
    block = np.zeros(((len(lines), block_size+1)), dtype=np.uint8)
    for line_idx, line in enumerate(lines):
        for chr_idx, chr in enumerate(line):
            if chr_idx >= block_size+1:
                break
            block[line_idx, chr_idx] = chr
    
    inputs = block[:, :-1]
    labels = block[:, 1:]
    return inputs, labels


@nnx.jit(static_argnums=(0, 1))
def create_sharded_model(_Model, dtype, rngs):
    model = _Model(rngs=rngs, dtype=dtype)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model


def load_sharded_model(fpath):
    rngs = nnx.Rngs(key)
    abstract_model = nnx.eval_shape(lambda: Model(rngs))
    graphdef, rngstate, other_state = nnx.split(abstract_model, nnx.RngState, ...)
    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(fpath, target=other_state)
    model = nnx.merge(graphdef, rngstate, restored_state)
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


def train():
    mesh = jax.sharding.Mesh(np.reshape(jax.devices(), (DATA_DIMS, MODEL_DIMS)),
                              ["data", "model"])

    print(mesh) 
    data_sharding = NamedSharding(mesh, PartitionSpec("data", None))


    with mesh:
        sharded_model = create_sharded_model(Model, DTYPE, rngs)

        total_params = count_params(sharded_model)
        print(f"Parameter Count: {total_params:,}")
        flops = 6 * (BATCH_SIZE * BLOCK_SIZE) * total_params
        print(f"FLOPS: {flops:,}")
        flops_per_device = flops / jax.device_count()
        print(f"FLOPS/device: {flops_per_device:,.4f}")


        tx = optax.adam(learning_rate=LEARNING_RATE)
        optimizer = nnx.Optimizer(sharded_model, tx)

        ds = tfds.load('lm1b', split='train', 
        shuffle_files=False)
        ds = ds.batch(BATCH_SIZE)
        ds = iter(ds)
        for step in range(4000):
            last_step_time = time.time()
            example = next(ds)
            #print(example['text'])
            inputs, labels = prepare_train_batch(example['text'].numpy(), BLOCK_SIZE)
            inputs = jax.device_put(inputs, data_sharding)
            labels = jax.device_put(labels, data_sharding)
            loss = step_fn(sharded_model, optimizer, inputs, labels)
            if step % 10 == 0:
                new_time = time.time()
                time_elapsed_seconds = (new_time - last_step_time)
                per_device_tflops_per_second = flops_per_device * 10 / 1e12 / time_elapsed_seconds
                print(f"step: {step}, loss: {loss}, time_elapsed: {time_elapsed_seconds}, tflops/device/s: {per_device_tflops_per_second}")
            step += 1

        _, _, other_state = nnx.split(sharded_model , nnx.RngState, ...)
        ckptr = ocp.StandardCheckpointer()
        path = Path().absolute() / "checkpoints" / "charformer_ckpt"
        shutil.rmtree(path, ignore_errors=True)
        ckptr.save(path, other_state)

        time.sleep(5)

        return sharded_model


#@nnx.jit(static_argnums=(2, 3)) # TODO: Fix inference time jit
def _pred(model, x, pos, use_cache):
    logits = model(x, pos=pos, use_cache=use_cache)
    preds = jnp.argmax(logits, axis=-1)
    return preds


def infer(model):
    text = np.array([b"I am a language model,"])
    len_text = len(text[0])
    x, _ = prepare_train_batch(text, len_text)
    preds = _pred(model, x, 0, True)
    print(chr(preds[0,-1]))
    pos = len_text
    for idx in range(20):
        x = preds[:, -1][..., None]
        #x = jnp.concatenate((x, preds[:, -1][..., None]), axis=-1)
        #print(x)
        preds = _pred(model, x, pos, True)
        print(chr(preds[0,-1]))
        pos += 1
    

if __name__ == "__main__":
    model = train()
    #w = model.layers[0].mlp.proj.kernel.value
    #jax.debug.visualize_array_sharding(w)

    #path = Path().absolute() / "checkpoints" / "charformer_ckpt"
    #model = load_sharded_model(path)
    #infer(model)