import jax
import jax.numpy as jnp
from flax import nnx
import optax

from flax.training import train_state 


def compute_global_norm(grads):
    return jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))


def loss_fn(model, batch, targets):
    logits = model(batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss



@nnx.jit(static_argnames=("B", "T"))
def accum_step(model, data, B, T, accum_grad, accum_loss, rng):
    k = jax.random.randint(rng, (1,), 0, len(data) - B*T - 1)[0]
    batch = jax.lax.dynamic_slice(data, (k,), (B*T,)).reshape((B, T))
    targets = jax.lax.dynamic_slice(data, (k+1,), (B*T,)).reshape((B, T))
    loss, grads =  nnx.value_and_grad(loss_fn)(model, batch, targets)
    accum_grad = jax.tree_util.tree_map(lambda x, y: x + y, accum_grad, grads)
    accum_loss = accum_loss + loss
    return accum_grad, accum_loss


@nnx.jit(static_argnames=("B", "T"))
def train_step(model, optimizer, data, B, T, rng):
    k = jax.random.randint(rng, (1,), 0, len(data) - B*T - 1)[0]
    batch = jax.lax.dynamic_slice(data, (k,), (B*T,)).reshape((B, T))
    targets = jax.lax.dynamic_slice(data, (k+1,), (B*T,)).reshape((B, T))
    loss, grads =  nnx.value_and_grad(loss_fn)(model, batch, targets)
    norm = compute_global_norm(grads)
    optimizer.update(grads)
    return loss, norm


