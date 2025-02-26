import jax
import jax.numpy as jnp
import flax.nnx as nnx

import tiktoken

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

#@nnx.jit(static_argnums=(3, 4)) # TODO: Implement jit for inference
def _generate(model, tokens, key, max_length=50, num_completions=5, temperature=0.7, top_k=50):
    tokens = jnp.expand_dims(tokens, axis=0)
    x = jnp.tile(tokens, (num_completions, 1)) 
    while x.shape[1] < max_length: 
        logits = model(x) 
        logits = logits[:, -1, :] / temperature # (B, vocab_size)
        x_next, key = top_k_sampling(logits, key, k=top_k)
        x_next = x_next.reshape(x_next.shape[0], 1)
        x = jnp.concatenate((x, x_next), axis=1) # (B, T+1)
    
    return x


def generate_completion(model: nnx.Module, prefix: str, *, max_length=50, num_completions=5, 
                        temperature=0.7, top_k = 50):

    key = jax.random.PRNGKey(0)    
    model.eval()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prefix)
    tokens = jnp.array(tokens, dtype=jnp.int32) 

    x = _generate(model, tokens, key, max_length, num_completions, temperature, top_k)

    for i in range(num_completions):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

