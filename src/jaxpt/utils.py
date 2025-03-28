import csv
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


def count_params(state: nnx.statelib.State) -> int:
    return len(list_params(state))


def list_params(state: nnx.statelib.State) -> list[str]:
    stack = [(k, k, state[k]) for k in state.keys()]

    params = []

    while len(stack) > 0:
        item_key, item_path, item_val = stack.pop()
        if type(item_val) is nnx.statelib.State:
            for k in item_val.keys():
                stack.append((k, f"{item_path}.{k}", item_val[k]))
        elif type(item_val) is nnx.variablelib.VariableState:
            if (
                (item_val.type != nnx.Variable)
                and (item_val.value is not None)
                and ("dropout" not in item_path)
            ):
                params.append(item_path)

    return params
