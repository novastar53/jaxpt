import os

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp

devices = jax.devices()
num_devices = len(devices)
print(f"num devices: {num_devices}")

from jax.sharding import Mesh, PartitionSpec, NamedSharding

mesh = Mesh(devices, ("devices",))
spec = PartitionSpec("devices",)
sharding = NamedSharding(mesh, spec)

x = jax.random.normal(jax.random.key(0), (num_devices, 2, num_devices, 3))
print(x)
x = jax.lax.with_sharding_constraint(x, sharding)
print(x.device)


x = jnp.swapaxes(x, 0, 2)
print(x.device)

x = jnp.reshape(x, (-1, 3))
x = jax.lax.with_sharding_constraint(x, sharding)
print(x.device)
x = jnp.reshape(x, (num_devices, 2*num_devices, 3))
print(x)


