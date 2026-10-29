"""Check free-running evaluation and repeated timing controls offline."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/experiments'))
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
import paired_kv_dataset as base
from train_cache_attention import stack_features
from evaluate_cache_attention import native_start, cached_start, continue_greedy, prefix_agreement, timed, lock_workdir, check_destination


class EvaluationTests(unittest.TestCase):
    def test_duplicate_runs_and_existing_destinations_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = lock_workdir(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, 'Another evaluation'):
                lock_workdir(Path(tmp))
            first.close()
            lock_workdir(Path(tmp)).close()
        class Api:
            def dataset_info(self, *args, **kwargs):
                return SimpleNamespace(private=True, siblings=[SimpleNamespace(rfilename='experiments/existing/evaluation.json')])
        with self.assertRaisesRegex(ValueError, 'already exists'):
            check_destination(Api(), 'experiments/existing')
        self.assertTrue(check_destination(Api(), 'experiments/new').private)

    def test_prefix_agreement_counts_first_divergence_and_eos(self):
        self.assertEqual(prefix_agreement([1, 2, 3], [1, 4, 3]),
                         {'first_token_match': True, 'matching_prefix_tokens': 1, 'sequence_equal': False})
        self.assertTrue(prefix_agreement([7], [7])['sequence_equal'])
        self.assertEqual(prefix_agreement([7], [7, 8])['matching_prefix_tokens'], 1)
        self.assertFalse(prefix_agreement([7], [7, 8])['sequence_equal'])

    def test_native_prefill_and_saved_cache_generate_identically(self):
        torch.manual_seed(3)
        torch.set_num_threads(2)
        c = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1)
        c._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(c).eval()
        tokens = torch.tensor([[3, 5, 7, 9, 2, 1]])
        features, shape = base.capture(model, tokens[0].tolist(), 'cpu')
        features = stack_features(features, shape)
        with torch.inference_mode():
            a = native_start(model, tokens)
            b = cached_start(model, features[:-1], tokens[:, -1:])
            torch.testing.assert_close(a.logits, b.logits, atol=1e-5, rtol=1e-4)
            self.assertEqual(continue_greedy(model, a, 6, set()), continue_greedy(model, b, 6, set()))
        count = []
        self.assertEqual(len(timed(lambda: count.append(1), 'cpu', 3)), 3)
        self.assertEqual(len(count), 4)  # One warm-up plus three timed runs.


if __name__ == '__main__':
    unittest.main()
