import numpy as np
import jax
import jax.numpy as jnp
import types
import pytest

from jaxpt.dataloaders import CharLoader, BaseDataLoader, DataLoader, CloudDataLoader, BlendedCloudDataLoader

class DummyBaseDataLoader(BaseDataLoader):
    def _list_shards(self, label):
        return ["dummy_shard_1", "dummy_shard_2"]
    def _load_shard(self):
        # Each shard is a 1D numpy array of 100 tokens
        return np.arange(100, dtype=np.uint16)

def test_charloader_encode_decode():
    text = "hello world"
    loader = CharLoader(text)
    encoded = loader.encode("hello")
    decoded = loader.decode(encoded)
    assert decoded == "hello"
    assert loader.vocab_size == len(set(text))

def test_charloader_get_batch():
    text = "abcdefghij"
    loader = CharLoader(text)
    data = loader.encode_text(text * 10)
    key = jax.random.PRNGKey(0)
    x, y = loader.get_batch(key, data, batch_size=2, block_size=5)
    assert x.shape == (2, 5)
    assert y.shape == (2, 5)
    print(x)
    print(y)
    # y should be x shifted by 1 within the data, not x + 1 numerically
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])

def test_basedataloader_call():
    loader = DummyBaseDataLoader(batch_size=2, block_size=5, device_rank=1)
    x, y = loader()
    assert x.shape == (1, 2, 5)
    assert y.shape == (1, 2, 5)
    assert np.allclose(y, x + 1)

def test_basedataloader_start_shard_pos():
    loader = DummyBaseDataLoader(batch_size=2, block_size=5, device_rank=1, start_shard=1, start_shard_pos=10)
    x, y = loader()
    assert x.shape == (1, 2, 5)
    assert y.shape == (1, 2, 5)

def test_dataloader(tmp_path):
    # Create dummy shards
    arr = np.arange(100, dtype=np.uint16)
    np.save(tmp_path / "shard1.npy", arr)
    np.save(tmp_path / "shard2.npy", arr + 100)
    loader = DataLoader(str(tmp_path), batch_size=2, block_size=5, device_rank=1)
    x, y = loader()
    assert x.shape == (1, 2, 5)
    assert y.shape == (1, 2, 5)

@pytest.mark.skip("Requires GCP credentials and bucket")
def test_clouddataloader(monkeypatch):
    # Mock Google Cloud Storage
    class DummyBlob:
        def __init__(self, name):
            self.name = name
        def download_as_bytes(self):
            arr = np.arange(100, dtype=np.uint16)
            import io
            buf = io.BytesIO()
            np.save(buf, arr)
            buf.seek(0)
            return buf.read()
    class DummyBucket:
        def __init__(self):
            self.blobs = [DummyBlob("shard1"), DummyBlob("shard2")]
        def list_blobs(self, prefix):
            return self.blobs
        def blob(self, name):
            return next(b for b in self.blobs if b.name == name)
    class DummyClient:
        def bucket(self, name):
            return DummyBucket()
    monkeypatch.setattr("google.cloud.storage.Client", DummyClient)
    loader = CloudDataLoader("bucket", "prefix", batch_size=2, block_size=5, device_rank=1)
    x, y = loader()
    assert x.shape == (1, 2, 5)
    assert y.shape == (1, 2, 5)

def test_blendedclouddataloader(monkeypatch):
    # Patch CloudDataLoader to return fixed arrays
    class DummyCloudDataLoader:
        def __init__(self, *args, **kwargs):
            self.calls = 0
        def __call__(self):
            x = np.ones((1, 2, 5), dtype=np.uint16) * self.calls
            y = x + 1
            self.calls += 1
            return x, y
    monkeypatch.setattr("jaxpt.dataloaders.CloudDataLoader", DummyCloudDataLoader)
    loader = BlendedCloudDataLoader(
        batch_size=4,
        block_size=5,
        bucket_names=["a", "b"],
        bucket_prefixes=["a", "b"],
        proportions=[1, 1],
        device_rank=1,
    )
    x, y = loader()
    assert x.shape == (1, 4, 5)
    assert y.shape == (1, 4, 5)
