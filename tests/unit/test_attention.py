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


@dataclass
class Attn_Config(Config):
    n_layer: int = 1
    block_size: int = 5  # sequence length
    n_head: int = 4  # number of attention heads
    n_kv_head: int = 2  # number of key-value heads
    n_embed: int = 12  # number token embedding dimensionsa
    sdpa_implementation: str = "slow"
    init_stddev: float = 0.02
    attention_bias: bool = False  # use bias in attention layers
    use_cache: bool = False
    rope_base_freq: float = 10000.0


@pytest.fixture
def rng_key():
    return jax.random.key(42)


def test_causal_self_attention():
    batch_size = 2
    config = Attn_Config(sdpa_implementation="xla")
    x = jax.random.normal(
        jax.random.key(0), (batch_size, config.block_size, config.n_embed)
    )
    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))
    y1 = attn(x)

    config = Attn_Config(sdpa_implementation="slow")
    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))
    y2 = attn(x)

    assert jnp.allclose(y1, y2, atol=1e-5)


def test_gq_attention():
    batch_size = 2
    config = Attn_Config(sdpa_implementation="xla")
    x = jax.random.normal(
        jax.random.key(0), (batch_size, config.block_size, config.n_embed)
    )
    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    y1 = attn(x)
    config = Attn_Config(sdpa_implementation="slow")
    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    y2 = attn(x)

    assert jnp.allclose(y1, y2, atol=1e-5)


def test_causal_self_attention_with_bias():
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="xla",
        attention_bias=True,
    )
    x = jax.random.normal(
        jax.random.key(0), (batch_size, config.block_size, config.n_embed)
    )
    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))
    y1 = attn(x)

    config = Attn_Config(
        sdpa_implementation="slow",
        attention_bias=True,
    )
    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))
    y2 = attn(x)

    assert jnp.allclose(y1, y2, atol=1e-5)


def test_causal_self_attention_with_mask(rng_key):
    batch_size = 2
    config = Attn_Config(sdpa_implementation="xla")
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )
    mask = jnp.ones((batch_size, config.block_size), dtype=jnp.bool_)

    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))
    y1 = attn(x, mask=mask)

    config = Attn_Config(sdpa_implementation="slow")
    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))
    y2 = attn(x, mask=mask)

    assert jnp.allclose(y1, y2, atol=1e-5)


def test_gq_attention_with_bias():
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="xla",
        attention_bias=True,
    )
    x = jax.random.normal(
        jax.random.key(0), (batch_size, config.block_size, config.n_embed)
    )
    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    y1 = attn(x)

    config = Attn_Config(
        sdpa_implementation="slow",
        attention_bias=True,
    )
    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    y2 = attn(x)

    assert jnp.allclose(y1, y2, atol=1e-5)


def test_gq_attention_with_mask(rng_key):
    batch_size = 2
    config = Attn_Config(sdpa_implementation="xla")
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )
    mask = jnp.ones((batch_size, config.block_size), dtype=jnp.bool_)

    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    y1 = attn(x, mask=mask)

    config = Attn_Config(sdpa_implementation="slow")
    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    y2 = attn(x, mask=mask)

    assert jnp.allclose(y1, y2, atol=1e-5)


def test_gq_attention_caching(rng_key):
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="slow",
        use_cache=True,
    )

    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn_cached = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    x1 = x[:, :3, :]
    y1 = attn_cached(x1)
    x2 = x[:, 3:, :]
    y2 = attn_cached(x2)

    assert y1.shape == (batch_size, 3, config.n_embed)
    assert y2.shape == (batch_size, 1, config.n_embed)


def test_gq_attention_caching_with_mask(rng_key):
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="slow",
        use_cache=True,
    )

    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )
    mask = jnp.ones((batch_size, config.block_size), dtype=jnp.bool_)

    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))

    y1 = attn(x[:, :3, :], mask=mask[:, :3])
    y2 = attn(x[:, 3:, :], mask=mask[:, 3:])

    assert y1.shape == (batch_size, 3, config.n_embed)
    assert y2.shape == (batch_size, 1, config.n_embed)


def test_causal_self_attention_rope(rng_key):
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="xla",
        n_embed=16,  # Even number to match RoPE dimensions
    )

    rope_omega = calc_rope_omega_llama(
        n_embed=config.n_embed,
        n_head=config.n_head,
        block_size=config.block_size,
        rope_base_freq=config.rope_base_freq,
        dtype=config.dtype,
    )

    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )
    attn = CausalSelfAttention_w_RoPE(
        config=config, rope_omega=rope_omega, rngs=nnx.Rngs(default=0)
    )
    y1 = attn(x)

    config = Attn_Config(
        sdpa_implementation="slow",
        n_embed=16,
    )
    rope_omega = calc_rope_omega_llama(
        n_embed=config.n_embed,
        n_head=config.n_head,
        block_size=config.block_size,
        rope_base_freq=config.rope_base_freq,
        dtype=config.dtype,
    )

    attn = CausalSelfAttention_w_RoPE(
        config=config, rope_omega=rope_omega, rngs=nnx.Rngs(default=0)
    )
    y2 = attn(x)

    assert jnp.allclose(y1, y2, atol=1e-5)


def test_gq_attention_rope(rng_key):
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="xla",
        n_embed=16,  # Even number to match RoPE dimensions
    )

    rope_omega = calc_rope_omega_llama(
        n_embed=config.n_embed,
        n_head=config.n_head,
        block_size=config.block_size,
        rope_base_freq=config.rope_base_freq,
        dtype=config.dtype,
    )

    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )
    attn = GQ_Attention_w_RoPE(
        config=config, rope_omega=rope_omega, rngs=nnx.Rngs(default=0)
    )
    y1 = attn(x)

    config = Attn_Config(
        sdpa_implementation="slow",
        n_embed=16,
    )
    rope_omega = calc_rope_omega_llama(
        n_embed=config.n_embed,
        n_head=config.n_head,
        block_size=config.block_size,
        rope_base_freq=config.rope_base_freq,
        dtype=config.dtype,
    )

    attn = GQ_Attention_w_RoPE(
        config=config, rope_omega=rope_omega, rngs=nnx.Rngs(default=0)
    )
    y2 = attn(x)

    assert jnp.allclose(y1, y2, atol=1e-5)


def test_gq_attention_rope_caching(rng_key):
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="slow",
        use_cache=True,
        n_embed=16,  # Even number to match RoPE dimensions
    )

    rope_omega = calc_rope_omega_llama(
        n_embed=config.n_embed,
        n_head=config.n_head,
        block_size=config.block_size,
        rope_base_freq=config.rope_base_freq,
        dtype=config.dtype,
    )

    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )
    attn = GQ_Attention_w_RoPE(
        config=config, rope_omega=rope_omega, rngs=nnx.Rngs(default=0)
    )

    x1 = x[:, :3, :]
    y1 = attn(x1)
    x2 = x[:, 3:, :]
    y2 = attn(x2)

    assert y1.shape == (batch_size, 3, config.n_embed)
    assert y2.shape == (batch_size, 2, config.n_embed)


def test_attention_different_sequence_lengths(rng_key):
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="slow",
    )

    x = jax.random.normal(rng_key, (batch_size, 3, config.n_embed))
    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))
    y = attn(x)

    assert y.shape == (batch_size, 3, config.n_embed)


def test_attention_single_batch(rng_key):
    config = Attn_Config(
        sdpa_implementation="slow",
    )

    x = jax.random.normal(rng_key, (1, config.block_size, config.n_embed))
    attn = CausalSelfAttention(config=config, rngs=nnx.Rngs(default=0))
    y = attn(x)

    assert y.shape == (1, config.block_size, config.n_embed)


def test_gq_attention_kv_head_division(rng_key):
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="slow",
        n_head=8,
        n_kv_head=4,
        n_embed=32,
    )

    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )
    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    y = attn(x)

    assert y.shape == (batch_size, config.block_size, config.n_embed)


def test_gq_attention_equal_kv_heads(rng_key):
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="slow",
        n_head=4,
        n_kv_head=4,
    )

    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )
    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    y = attn(x)

    assert y.shape == (batch_size, config.block_size, config.n_embed)


def test_causal_self_attention_deterministic(rng_key):
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="slow",
    )

    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    key1 = nnx.Rngs(default=0)
    attn1 = CausalSelfAttention(config=config, rngs=key1)
    y1 = attn1(x)

    key2 = nnx.Rngs(default=0)
    attn2 = CausalSelfAttention(config=config, rngs=key2)
    y2 = attn2(x)

    assert jnp.allclose(y1, y2)


def test_gq_attention_deterministic(rng_key):
    batch_size = 2
    config = Attn_Config(
        sdpa_implementation="slow",
    )

    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    key1 = nnx.Rngs(default=0)
    attn1 = GQ_Attention(config=config, rngs=key1)
    y1 = attn1(x)

    key2 = nnx.Rngs(default=0)
    attn2 = GQ_Attention(config=config, rngs=key2)
    y2 = attn2(x)

    assert jnp.allclose(y1, y2)


def test_kv_cache_storage(rng_key):
    config = Attn_Config(
        sdpa_implementation="slow",
        use_cache=True,
    )

    batch_size = 2
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))

    assert attn.key_cache is None
    assert attn.value_cache is None

    _ = attn(x[:, :3, :])

    assert attn.key_cache is not None
    assert attn.value_cache is not None
    assert attn.key_cache.shape == (
        batch_size,
        3,
        config.n_kv_head * config.n_embed // config.n_head,
    )
    assert attn.value_cache.shape == (
        batch_size,
        3,
        config.n_kv_head * config.n_embed // config.n_head,
    )


def test_kv_cache_correctness(rng_key):
    config = Attn_Config(
        sdpa_implementation="slow",
        use_cache=False,
    )

    batch_size = 2
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn_no_cache = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))
    y_no_cache = attn_no_cache(x)

    config.use_cache = True
    attn_with_cache = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))

    y_cached_parts = []
    for i in range(config.block_size):
        y = attn_with_cache(x[:, i : i + 1, :])
        y_cached_parts.append(y)

    y_cached = jnp.concatenate(y_cached_parts, axis=1)

    assert jnp.allclose(y_no_cache[:, :-1, :], y_cached[:, :-1, :], atol=1e-5)


def test_kv_cache_truncation(rng_key):
    config = Attn_Config(
        sdpa_implementation="slow",
        use_cache=True,
        block_size=10,
    )

    batch_size = 2
    x = jax.random.normal(rng_key, (batch_size, 15, config.n_embed))

    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))

    for i in range(15):
        attn(x[:, i : i + 1, :])

    assert attn.key_cache.shape[1] <= config.block_size
    assert attn.value_cache.shape[1] <= config.block_size
    assert attn.key_cache.shape == (
        batch_size,
        config.block_size,
        config.n_kv_head * config.n_embed // config.n_head,
    )
    assert attn.value_cache.shape == (
        batch_size,
        config.block_size,
        config.n_kv_head * config.n_embed // config.n_head,
    )


def test_kv_cache_with_rope(rng_key):
    config = Attn_Config(
        sdpa_implementation="slow",
        use_cache=True,
        n_embed=16,
    )

    rope_omega = calc_rope_omega_llama(
        n_embed=config.n_embed,
        n_head=config.n_head,
        block_size=config.block_size,
        rope_base_freq=config.rope_base_freq,
        dtype=config.dtype,
    )

    batch_size = 2
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn = GQ_Attention_w_RoPE(
        config=config, rope_omega=rope_omega, rngs=nnx.Rngs(default=0)
    )

    assert attn.key_cache is None
    assert attn.value_cache is None

    _ = attn(x[:, :3, :])

    cache_len = attn.key_cache.shape[1]
    assert cache_len == 3
    head_dim = config.n_embed // config.n_head
    assert attn.key_cache.shape == (batch_size, 3, config.n_kv_head, head_dim)
    assert attn.value_cache.shape == (batch_size, 3, config.n_kv_head, head_dim)

    _ = attn(x[:, 3:, :])

    assert attn.key_cache.shape[1] == config.block_size


def test_kv_cache_single_token_generation(rng_key):
    config = Attn_Config(
        sdpa_implementation="slow",
        use_cache=True,
    )

    batch_size = 2
    x = jax.random.normal(
        rng_key, (batch_size, config.block_size, config.n_embed)
    )

    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))

    y_full = attn(x)

    attn_cache = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))

    y_cache_parts = []
    for i in range(config.block_size):
        y = attn_cache(x[:, i : i + 1, :])
        y_cache_parts.append(y)

    y_cached = jnp.concatenate(y_cache_parts, axis=1)

    assert y_cached.shape == (batch_size, config.block_size, config.n_embed)
    assert jnp.allclose(y_full[:, :-1, :], y_cached[:, :-1, :], atol=1e-5)


def test_kv_cache_state_persistence(rng_key):
    config = Attn_Config(
        sdpa_implementation="slow",
        use_cache=True,
        block_size=10,
    )

    batch_size = 2
    x = jax.random.normal(rng_key, (batch_size, 10, config.n_embed))

    attn = GQ_Attention(config=config, rngs=nnx.Rngs(default=0))

    _ = attn(x[:, :3, :])
    cache_after_first = attn.key_cache.shape[1]

    _ = attn(x[:, 3:6, :])
    cache_after_second = attn.key_cache.shape[1]

    assert cache_after_second > cache_after_first
    assert cache_after_second == 6
