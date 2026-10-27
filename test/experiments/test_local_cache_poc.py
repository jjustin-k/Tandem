# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2",
#   "numpy==2.3.4", "psutil==7.1.2",
# ]
# ///
"""Offline tests: no pretrained weights or tokenizers are downloaded."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

SCRIPT = Path(__file__).resolve().parents[2] / "src/experiments/local_cache_poc.py"
spec = importlib.util.spec_from_file_location("poc", SCRIPT)
poc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poc)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, max_position_embeddings=128)
        config._attn_implementation = "eager"
        self.model = Qwen2ForCausalLM(config).eval()
        self.prefix, self.suffix = [10, 11, 12, 13, 14], [15, 16, 17]
        native = poc.forward(self.model, self.prefix, "cpu")
        self.tensors = poc.tensors_from_cache(native.past_key_values)
        self.identity = poc.identity_for(self.prefix)

    def test_serialized_replay_matches_native_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.safetensors"
            poc.save_cache(path, self.tensors, self.identity)
            cached = poc.restore_cache(path, self.identity, self.model.config, "cpu", torch.float32)
            actual = poc.forward(self.model, self.suffix, "cpu", cached)
            native = poc.forward(self.model, self.prefix + self.suffix, "cpu")
            torch.testing.assert_close(actual.logits, native.logits, atol=1e-5, rtol=1e-5)
            self.assertEqual(actual.past_key_values.get_seq_length(), 8)
            self.assertEqual(self.tensors["0.k"].shape[-2], 5)
            self.assertEqual(poc.complete(self.model, actual, "cpu", 8, set()),
                             poc.complete(self.model, native, "cpu", 8, set()))

    def test_wrong_prefix_and_model_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.safetensors"
            poc.save_cache(path, self.tensors, self.identity)
            for changed in (dict(self.identity, token_ids_sha256="other"),
                            dict(self.identity, model="other")):
                with self.assertRaisesRegex(ValueError, "identity"):
                    poc.restore_cache(path, changed, self.model.config, "cpu", torch.float32)

    def test_truncated_missing_nonfinite_and_wrong_dtype_are_rejected(self):
        for mode in ("truncated", "missing", "nonfinite", "dtype"):
            tensors = {k: v.clone() for k, v in self.tensors.items()}
            if mode == "truncated":
                tensors["0.k"] = tensors["0.k"][..., :-1, :]
            elif mode == "missing":
                del tensors["0.k"]
            elif mode == "nonfinite":
                tensors["0.k"][0, 0, 0, 0] = float("nan")
            else:
                tensors["0.k"] = tensors["0.k"].half()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                poc.validate_tensors(tensors, self.model.config, 5, torch.float32)

    def test_fresh_cache_is_used_for_each_trial(self):
        first = poc.cache_from_tensors(self.tensors, self.model.config, "cpu")
        poc.forward(self.model, self.suffix, "cpu", first)
        second = poc.cache_from_tensors(self.tensors, self.model.config, "cpu")
        self.assertEqual(first.get_seq_length(), 8)
        self.assertEqual(second.get_seq_length(), 5)


if __name__ == "__main__":
    unittest.main()
