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


@nnx.pmap(in_axes=(None, 0, 0), out_axes=(0, 0))
def train_step(model, batch, targets):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch, targets)
    return loss, grads

