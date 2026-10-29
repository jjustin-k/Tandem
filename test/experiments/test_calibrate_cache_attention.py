"""Continuation alignment and gradients, without model or dataset downloads."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/experiments'))
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
import paired_kv_dataset as base
from train_cache_attention import stack_features, receiver_logits, logit_kl
from calibrate_cache_attention import continuation_inputs, teacher_logits


class ContinuationTests(unittest.TestCase):
    def test_continuation_alignment_including_one_token_and_eos(self):
        prompt = torch.tensor([[3, 5, 7]])
        for continuation in ([9], [9, 11, 2]):
            full, suffix = continuation_inputs(prompt, continuation)
            self.assertEqual(full.tolist(), [[3, 5, 7] + list(continuation[:-1])])
            self.assertEqual(suffix.tolist(), [[7] + list(continuation[:-1])])
            self.assertEqual(suffix.shape[1], len(continuation))
        with self.assertRaises(ValueError):
            continuation_inputs(prompt, [])

    def test_native_continuation_control_and_frozen_receiver_gradients(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
        config._attn_implementation = 'eager'
        receiver = Qwen2ForCausalLM(config).eval().requires_grad_(False)
        prompt = torch.tensor([[3, 5, 7, 9, 2, 1]])
        tensors, geometry = base.capture(receiver, prompt[0].tolist(), 'cpu')
        features = stack_features(tensors, geometry)[:-1]
        continuation = [4, 6, 8, 2]
        teacher = teacher_logits(receiver, prompt, continuation)
        _, suffix = continuation_inputs(prompt, continuation)
        control = receiver_logits(receiver, features, suffix)
        torch.testing.assert_close(control, teacher, atol=1e-5, rtol=1e-4)
        self.assertFalse(teacher.requires_grad)
        translated = (features + .15 * torch.randn_like(features)).requires_grad_()
        loss = logit_kl(receiver_logits(receiver, translated, suffix), teacher)
        loss.backward()
        self.assertTrue(torch.isfinite(translated.grad).all())
        self.assertGreater(translated.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in receiver.parameters()))
        # Later continuation positions must also teach the translated prefix.
        translated.grad = None
        logit_kl(receiver_logits(receiver, translated, suffix)[:, 1:], teacher[:, 1:]).backward()
        self.assertGreater(translated.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
