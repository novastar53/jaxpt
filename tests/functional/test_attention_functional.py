from dataclasses import dataclass
import pytest

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from jaxpt.modules.attention import (
    CausalSelfAttention,
    GQ_Attention,
    CausalSelfAttention_w_RoPE,
    GQ_Attention_w_RoPE,
)
from jaxpt.modules.config import Config
from jaxpt.modules.position import calc_rope_omega_llama


@pytest.fixture
def rng_key():
    return jax.random.key(42)


@dataclass
class Attn_Config(Config):
    n_layer: int = 1
    block_size: int = 128  # Larger sequence length for functional tests
    n_head: int = 8  # More heads for realistic scenario
    n_kv_head: int = 4  # Grouped query attention
    n_embed: int = 256  # Larger embedding for realistic scenario
    sdpa_implementation: str = "slow"
    init_stddev: float = 0.02
    attention_bias: bool = False
    use_cache: bool = False
    rope_base_freq: float = 10000.0


def test_causal_self_attention_forward_pass(rng_key):
    config = Attn_Config(
        block_size=64,
        n_head=4,
        n_embed=128,
    )

    batch_size = 4
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))
    y = attn(x)

    assert y.shape == (batch_size, config.block_size, config.n_embed)
    assert not jnp.any(jnp.isnan(y))


def test_gq_attention_forward_pass(rng_key):
    config = Attn_Config(
        block_size=64,
        n_head=8,
        n_kv_head=4,
        n_embed=256,
    )

    batch_size = 4
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    y = attn(x)

    assert y.shape == (batch_size, config.block_size, config.n_embed)
    assert not jnp.any(jnp.isnan(y))


def test_attention_autoregressive_generation(rng_key):
    config = Attn_Config(
        block_size=64,
        n_head=4,
        n_embed=128,
        use_cache=True,
    )

    batch_size = 2
    sequence_length = 16

    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))

    x = jax.random.normal(rng_key, (batch_size, 1, config.n_embed))
    outputs = []

    for _ in range(sequence_length):
        y = attn(x)
        outputs.append(y)
        x = y

    outputs = jnp.concatenate(outputs, axis=1)
    assert outputs.shape == (batch_size, sequence_length, config.n_embed)
    assert not jnp.any(jnp.isnan(outputs))


def test_attention_with_rope_positional_encoding(rng_key):
    config = Attn_Config(
        block_size=64,
        n_head=4,
        n_embed=128,
    )

    rope_omega = calc_rope_omega_llama(
        n_embed=config.n_embed,
        n_head=config.n_head,
        block_size=config.block_size,
        rope_base_freq=config.rope_base_freq,
        dtype=config.dtype,
    )

    batch_size = 4
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn = CausalSelfAttention_w_RoPE(
        config=config, rope_omega=rope_omega, rngs=nnx.Rngs(default=0)
    )
    y = attn(x)

    assert y.shape == (batch_size, config.block_size, config.n_embed)
    assert not jnp.any(jnp.isnan(y))


def test_gq_attention_with_rope_forward_pass(rng_key):
    config = Attn_Config(
        block_size=64,
        n_head=8,
        n_kv_head=4,
        n_embed=128,
    )

    rope_omega = calc_rope_omega_llama(
        n_embed=config.n_embed,
        n_head=config.n_head,
        block_size=config.block_size,
        rope_base_freq=config.rope_base_freq,
        dtype=config.dtype,
    )

    batch_size = 4
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn = GQ_Attention_w_RoPE(
        config=config, rope_omega=rope_omega, rngs=nnx.Rngs(default=0)
    )
    y = attn(x)

    assert y.shape == (batch_size, config.block_size, config.n_embed)
    assert not jnp.any(jnp.isnan(y))


def test_attention_batch_consistency(rng_key):
    config = Attn_Config(
        block_size=32,
        n_head=4,
        n_embed=128,
    )

    batch_size = 8
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))
    y = attn(x)

    assert y.shape == (batch_size, config.block_size, config.n_embed)

    y0 = attn(x[0:1, :, :])
    assert jnp.allclose(y[0], y0[0])


def test_attention_caching_consistency(rng_key):
    config = Attn_Config(
        block_size=32,
        n_head=4,
        n_embed=128,
    )

    batch_size = 2
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn_with_cache = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    attn_with_cache.config.use_cache = True

    y_cached_parts = []
    for i in range(config.block_size):
        y = attn_with_cache(x[:, i : i + 1, :])
        y_cached_parts.append(y)

    y_cached = jnp.concatenate(y_cached_parts, axis=1)

    assert y_cached.shape == (batch_size, config.block_size, config.n_embed)


def test_attention_gradient_flow(rng_key):
    config = Attn_Config(
        block_size=16,
        n_head=4,
        n_embed=64,
    )

    batch_size = 2
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))

    def loss_fn(attn, x):
        y = attn(x)
        return jnp.mean(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(attn, x)

    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()
    assert not jnp.isnan(loss)

    assert grads is not None
    assert len(grads) > 0


def test_attention_multiple_implementations_consistency(rng_key):
    config = Attn_Config(
        block_size=16,
        n_head=4,
        n_embed=64,
    )

    batch_size = 2
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    implementations = ["xla", "slow"]
    outputs = []

    for impl in implementations:
        config.sdpa_implementation = impl
        attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))
        y = attn(x)
        outputs.append(y)

    for i in range(1, len(outputs)):
        assert jnp.allclose(outputs[0], outputs[i], atol=1e-5)


def test_attention_with_padding_mask(rng_key):
    config = Attn_Config(
        block_size=16,
        n_head=4,
        n_embed=64,
    )

    batch_size = 3
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))

    mask = jnp.ones((batch_size, config.block_size), dtype=jnp.bool_)
    mask = mask.at[0, 10:].set(False)
    mask = mask.at[1, 12:].set(False)

    y = attn(x, mask=mask)

    assert y.shape == (batch_size, config.block_size, config.n_embed)
    assert not jnp.any(jnp.isnan(y))


def test_attention_variable_sequence_lengths(rng_key):
    config = Attn_Config(
        block_size=32,
        n_head=4,
        n_embed=64,
    )

    x_short = jax.random.normal(rng_key, (2, 8, config.n_embed))
    x_long = jax.random.normal(rng_key, (2, 16, config.n_embed))

    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))

    y_short = attn(x_short)
    y_long = attn(x_long)

    assert y_short.shape == (2, 8, config.n_embed)
    assert y_long.shape == (2, 16, config.n_embed)

    assert not jnp.any(jnp.isnan(y_short))
    assert not jnp.any(jnp.isnan(y_long))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
