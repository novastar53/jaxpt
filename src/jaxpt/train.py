import jax
import jax.numpy as jnp
from flax import nnx
import optax


def compute_global_norm(grads):
    return jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))


def loss_fn(model, batch, targets):
    logits = model(batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss


@nnx.jit
def train_step(model, optimizer, batch, targets):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch, targets)
    optimizer.update(grads)
    return loss, grads


@nnx.pmap(axis_name="devices", in_axes=(None, None, 0, 0), out_axes=(0, 0))
def parallel_train_step(model, optimizer, batch, targets):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch, targets)
    loss = jax.lax.pmean(loss, axis_name="devices")
    grads = jax.lax.pmean(grads, axis_name="devices")
    optimizer.update(grads)
    return loss, grads


@nnx.pmap(axis_name="devices", in_axes=(None, None, 0, 0), out_axes=(0, 0, 0))
def accum_train_step(model, optimizer, batches, targets):
    accum_grads = None
    accum_loss = 0
    for batch, target in zip(batches, targets):
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch, target)
        loss = jax.lax.pmean(loss, axis_name="devices")
        grads = jax.lax.pmean(grads, axis_name="devices")
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = jax.tree_util.tree_map(lambda x, y: x + y, accum_grads, grads)
        accum_loss += loss
    avg_grads = jax.tree_util.tree_map(lambda x: x / len(batches), accum_grads)
    optimizer.update(avg_grads)
    avg_loss = accum_loss / len(batches)
    norm = compute_global_norm(avg_grads)
    return avg_loss, avg_grads, norm
