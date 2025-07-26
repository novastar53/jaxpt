import csv
from english_words import get_english_words_set
import random

import jax
import jax.numpy as jnp
from flax import nnx

# Checkpointing 

def generate_random_code(length=6):
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def generate_readable_code():
    words = [w.lower() for w in get_english_words_set(['web2']) if 4 <= len(w) <= 8]
    return f"{random.choice(words)}_{random.choice(words)}"

# Logging


def append_to_csv(file_path, row):
    """
    Appends a single row to a CSV file.

    :param file_path: Path to the CSV file.
    :param row: A list or tuple representing the row to append.
    """
    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)


# Model Loaders


@nnx.jit(static_argnums=(0, 1))
def create_sharded_model(Model, config, rngs):
    model = Model(config=config, rngs=rngs)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = nnx.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model


# Model State PyTree Manipulation


def get_param(
    state: nnx.statelib.State, path: str
) -> nnx.variablelib.VariableState:
    keys = path.split(".")
    param = state
    for key in keys:
        if all(char.isdigit() for char in key):
            key = int(key)
        param = param[key]
    assert type(param) is nnx.variablelib.VariableState
    return param


def update_param(
    state: nnx.statelib.State, path: str, value: jnp.array
) -> nnx.variablelib.VariableState:
    param = get_param(state, path)
    assert param.value.shape == value.shape
    param.value = jnp.array(value)
    test_param = get_param(state, path)
    assert jnp.sum(test_param.value) == jnp.sum(value)
    return state


def count_params(m: nnx.Module, layer_type: str | None = None) -> int:
    def get_size(y):
        return y.size
    
    if layer_type is not None:
        def _filter(path, val):
            return issubclass(val.type, nnx.Param) and layer_type in path
        _, params, _ = nnx.split(m, _filter, nnx.Variable)
    else:
        _, params, _ = nnx.split(m, nnx.Param, nnx.Variable)
    param_counts = jax.tree_util.tree_map(get_size, params)
    total_params = jax.tree_util.tree_reduce(
        lambda x, y: x + y, param_counts, 0
    )

    return total_params

