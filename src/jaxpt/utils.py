import csv

import jax
import jax.numpy as jnp
from flax import nnx

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


# Model State PyTree Manipulation

def get_param(state: nnx.statelib.State, path: str) -> nnx.variablelib.VariableState:
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


def count_params(m: nnx.Module) -> int:

    def get_size(y):
        return y.size

    _, params, _ = nnx.split(m, nnx.Param, nnx.Variable)
    param_counts = jax.tree_util.tree_map(get_size, params)
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y, param_counts, 0)

    return total_params

