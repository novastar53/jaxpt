import jax
from flax import nnx
import optax

from flax.training import train_state 

@nnx.jit
def loss_fn(model, batch, targets):
    logits = model(batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss


@nnx.jit
def train_step(model, optimizer, batch, targets):
   
    loss, grads =  nnx.value_and_grad(loss_fn)(model, batch, targets)
    optimizer.update(grads)
    return loss

