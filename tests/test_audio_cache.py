import tempfile
from pathlib import Path
import torch

from automir.audio.cache import FeatureCache


def test_feature_cache_deterministic_hashing():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = FeatureCache(cache_dir=tmpdir, enabled=True)

        params_1 = {"rep": "logmel", "n_mels": 96, "sr": 22050}
        params_2 = {"sr": 22050, "n_mels": 96, "rep": "logmel"}  # same params, different order
        params_3 = {"rep": "logmel", "n_mels": 128, "sr": 22050}  # different params

        key_1 = cache.generate_cache_key("audio_01", params_1)
        key_2 = cache.generate_cache_key("audio_01", params_2)
        key_3 = cache.generate_cache_key("audio_01", params_3)

        assert key_1 == key_2, "Same parameters in different order must produce identical hash!"
        assert key_1 != key_3, "Different parameters must produce distinct hash to prevent stale features!"


def test_feature_cache_put_get_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = FeatureCache(cache_dir=tmpdir, enabled=True)
        params = {"rep": "logmel", "n_mels": 96}
        dummy_tensor = torch.randn(1, 96, 100)

        # Before putting
        assert cache.get("audio_test", params) is None

        # Put
        cache.put("audio_test", params, dummy_tensor)
        retrieved = cache.get("audio_test", params)
        assert retrieved is not None
        assert torch.allclose(dummy_tensor, retrieved)

        # Clear
        deleted = cache.clear()
        assert deleted == 1
        assert cache.get("audio_test", params) is None
