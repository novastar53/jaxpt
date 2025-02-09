import jax
from flax import nnx
import optax

from flax.training import train_state 


def train_step(model, optimizer, batch, targets):
    def loss_fn(model):
        logits = model(batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss
    
    loss, grads =  nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


