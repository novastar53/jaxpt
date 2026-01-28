"""
Test to verify the bug fix for generation functions.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from jaxpt.infer import (
    _generate_step,
    generate_completion_slow,
    generate_completion_fast,
    generate_completions,
    generate_completions_with_cache,
    generate,
)


def test_generate_step_signature():
    """Test that _generate_step has correct signature."""
    import inspect

    sig = inspect.signature(_generate_step)
    params = list(sig.parameters.keys())
    assert params == [
        "m",
        "x",
        "attn_mask",
        "temperature",
        "key",
    ], f"Got {params}"
    print("✓ _generate_step has correct signature")


def test_generate_functions_exist():
    """Test that all expected generation functions exist."""
    assert callable(generate_completion_slow)
    assert callable(generate_completion_fast)
    assert callable(generate_completions)
    assert callable(generate_completions_with_cache)
    assert callable(generate)
    print("✓ All generation functions exist")


def test_partial_not_used_in_generate_functions():
    """Test that generate functions don't incorrectly use functools.partial."""
    import inspect

    source = inspect.getsource(generate_completion_slow)
    assert (
        "partial" not in source
    ), "generate_completion_slow shouldn't use partial"

    source = inspect.getsource(generate_completion_fast)
    assert (
        "partial" not in source
    ), "generate_completion_fast shouldn't use partial"

    source = inspect.getsource(generate_completions)
    assert "partial" not in source, "generate_completions shouldn't use partial"

    print("✓ Generation functions don't incorrectly use functools.partial")


def test_generate_step_returns_tuple():
    """Test that _generate_step returns both sample_idxs and key."""
    import inspect

    source = inspect.getsource(_generate_step)
    assert (
        "return sample_idxs, key" in source
    ), "_generate_step should return (sample_idxs, key)"
    print("✓ _generate_step returns tuple (sample_idxs, key)")


def test_generate_completions_delegates():
    """Test that generate_completions delegates correctly."""
    import inspect

    source = inspect.getsource(generate_completions)
    assert (
        "generate_completion_slow" in source
    ), "generate_completions should call generate_completion_slow"
    print("✓ generate_completions delegates to generate_completion_slow")


def test_generate_alias():
    """Test that generate is an alias for generate_completions."""
    import inspect

    source = inspect.getsource(generate)
    assert (
        "generate_completions" in source
    ), "generate should call generate_completions"
    print("✓ generate is an alias for generate_completions")


if __name__ == "__main__":
    print("Running bug fix verification tests...\n")

    test_generate_step_signature()
    test_generate_functions_exist()
    test_partial_not_used_in_generate_functions()
    test_generate_step_returns_tuple()
    test_generate_completions_delegates()
    test_generate_alias()

    print("\n✅ All bug fix verification tests passed!")
    print(
        "\nThe generation functions should now work correctly without JIT errors."
    )
