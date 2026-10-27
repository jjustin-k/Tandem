# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2"]
# ///
"""Same-model KV replay control, not a cross-model transfer result.

uv run --script src/experiments/local_cache_poc.py --test
uv run --script src/experiments/local_cache_poc.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, Qwen2Config, Qwen2ForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ATOL = RTOL = 1e-3


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def cache_tensors(cache):
    return {f"{i}.{role}": tensor.detach().cpu().contiguous()
            for i, layer in enumerate(cache.layers)
            for role, tensor in (("k", layer.keys), ("v", layer.values))}


def validate(tensors, config, length):
    head = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    shape = (1, config.num_key_value_heads, length, head)
    names = {f"{i}.{role}" for i in range(config.num_hidden_layers) for role in ("k", "v")}
    if set(tensors) != names:
        raise ValueError("Missing or unexpected cache layers")
    for name, tensor in tensors.items():
        bad = tuple(tensor.shape) != shape or tensor.dtype != torch.float32
        if bad or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid cache: {name}")


def to_cache(tensors, config, device):
    cache = DynamicCache(config=config)
    for i in range(config.num_hidden_layers):
        # Clone: decoding appends to the cache in place.
        cache.update(tensors[f"{i}.k"].to(device).clone(), tensors[f"{i}.v"].to(device).clone(), i)
    return cache


def identity(prefix):
    digest = hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
    return {"model": MODEL, "revision": REVISION, "n": len(prefix), "tokens": digest}


def save_cache(path, tensors, ident):
    save_file(tensors, str(path), metadata={"identity": json.dumps(ident, sort_keys=True)})


def restore_cache(path, ident, config, device):
    with safe_open(str(path), framework="pt", device="cpu") as stored:
        if json.loads(stored.metadata()["identity"]) != ident:
            raise ValueError("Cache identity does not match model or prefix")
    tensors = load_file(str(path))
    validate(tensors, config, ident["n"])
    return to_cache(tensors, config, device)


def prompt(tokenizer, length, position, seed):
    code = "".join(random.Random(seed).choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=4))
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    header = enc("<|im_start|>system\nAnswer using only the provided project notes. "
                 "If the answer is absent, say UNKNOWN.<|im_end|>\n<|im_start|>user\n")
    fact = enc(f"\nProject Cedar's release code is {code}.\n")
    filler = enc("The archive contains routine notes about schedules, meetings, "
                 "testing, and documentation. These notes contain no release codes.\n")
    room = length - len(header) - len(fact)
    if room < 0:
        raise ValueError("prefix shorter than the fact")
    before = {"early": 0, "middle": room // 2, "late": room}[position]
    filler *= length // len(filler) + 2
    prefix = header + filler[:before] + fact + filler[:room - before]
    question = enc("\nWhat is Project Cedar's release code? Answer with the code only."
                   "<|im_end|>\n<|im_start|>assistant\n")
    return prefix, question, header + question, code


@torch.inference_mode()
def forward(model, tokens, device, cache=None):
    past = 0 if cache is None else cache.get_seq_length()
    index = torch.arange(past, past + len(tokens), device=device)
    return model(input_ids=torch.tensor([tokens], device=device),
                 attention_mask=torch.ones(1, past + len(tokens), dtype=torch.long, device=device),
                 position_ids=index[None, :], cache_position=index,
                 past_key_values=cache, use_cache=True, logits_to_keep=1)


@torch.inference_mode()
def complete(model, first, device, limit, eos):
    output, tokens = first, []
    for step in range(limit):
        token = int(output.logits[0, -1].argmax())
        tokens.append(token)
        if token in eos or step == limit - 1:
            break
        output = forward(model, [token], device, output.past_key_values)
    return tokens


def run(args):
    auto = "mps" if torch.backends.mps.is_available() else "cpu"
    device = auto if args.device == "auto" else args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; pass --device cpu")
    if args.repeats < 1 or args.max_new_tokens < 1:
        raise ValueError("repeats and max-new-tokens must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager").eval().to(device)
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    forward(model, [tokenizer.eos_token_id], device)
    sync(device)
    hits = blank = cases = 0
    for length in args.lengths:
        for index, position in enumerate(("early", "middle", "late")):
            prefix, question, bare, code = prompt(tokenizer, length, position, args.seed + index + length)
            ident = identity(prefix)
            tensors = cache_tensors(forward(model, prefix, device).past_key_values)
            validate(tensors, model.config, length)
            path = args.output / "cache.safetensors"
            save_cache(path, tensors, ident)
            if not all(torch.equal(load_file(str(path))[key], value) for key, value in tensors.items()):
                raise AssertionError("serialization changed a tensor")
            seen, medians = {}, {}
            for arm in ("native", "warm", "restored"):
                samples = []
                for rep in range(args.repeats):
                    cache = to_cache(tensors, model.config, device) if arm == "warm" else None
                    sync(device)
                    start = time.perf_counter()
                    if arm == "restored":
                        cache = restore_cache(path, ident, model.config, device)
                    tokens = prefix + question if arm == "native" else question
                    output = forward(model, tokens, device, cache)
                    int(output.logits[0, -1].argmax())
                    sync(device)
                    samples.append((time.perf_counter() - start) * 1000)
                    if output.past_key_values.get_seq_length() != length + len(question):
                        raise AssertionError("cache length")
                    if rep == 0:
                        generated = complete(model, output, device, args.max_new_tokens, eos)
                        seen[arm] = output.logits[0, -1].float().cpu(), generated
                medians[arm] = statistics.median(samples)
            native_logits, native_tokens = seen["native"]
            for arm in ("warm", "restored"):
                logits, generated = seen[arm]
                same = generated == native_tokens and torch.allclose(native_logits, logits, atol=ATOL, rtol=RTOL)
                if not same:
                    raise AssertionError(f"{arm} replay diverged")
            answer = tokenizer.decode(native_tokens, skip_special_tokens=True).strip()
            empty_ids = complete(model, forward(model, bare, device), device, args.max_new_tokens, eos)
            empty = tokenizer.decode(empty_ids, skip_special_tokens=True).strip()
            hits += answer == code
            blank += empty == code
            cases += 1
            path.unlink()
            print(f"{length}/{position}: {answer!r} expected {code!r}  "
                  f"native {medians['native']:.0f}ms  warm {medians['warm']:.0f}ms  "
                  f"restored {medians['restored']:.0f}ms", flush=True)
            if device == "mps":
                torch.mps.empty_cache()
    passed = hits >= math.ceil(8 * cases / 9) and blank == 0
    print(f"facts {hits}/{cases}; no-context {blank}; passed {passed}")
    return 0 if passed else 1


class ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=128)
        config._attn_implementation = "eager"
        self.model = Qwen2ForCausalLM(config).eval()
        self.prefix, self.suffix = [10, 11, 12, 13, 14], [15, 16, 17]
        native = forward(self.model, self.prefix, "cpu")
        self.tensors = cache_tensors(native.past_key_values)
        self.ident = identity(self.prefix)

    def test_replay_matches_native(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.safetensors"
            save_cache(path, self.tensors, self.ident)
            actual = forward(self.model, self.suffix, "cpu",
                             restore_cache(path, self.ident, self.model.config, "cpu"))
            native = forward(self.model, self.prefix + self.suffix, "cpu")
            torch.testing.assert_close(actual.logits, native.logits, atol=1e-5, rtol=1e-5)
            self.assertEqual(complete(self.model, actual, "cpu", 8, set()),
                             complete(self.model, native, "cpu", 8, set()))
            self.assertEqual(self.tensors["0.k"].shape[-2], 5)

    def test_bad_identity_and_tensors_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.safetensors"
            save_cache(path, self.tensors, self.ident)
            with self.assertRaisesRegex(ValueError, "identity"):
                restore_cache(path, dict(self.ident, model="other"), self.model.config, "cpu")
        for mode in ("truncated", "missing", "nonfinite", "dtype"):
            tensors = {key: value.clone() for key, value in self.tensors.items()}
            if mode == "truncated":
                tensors["0.k"] = tensors["0.k"][..., :-1, :]
            elif mode == "missing":
                del tensors["0.k"]
            elif mode == "nonfinite":
                tensors["0.k"][0, 0, 0, 0] = float("nan")
            else:
                tensors["0.k"] = tensors["0.k"].half()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate(tensors, self.model.config, 5)

    def test_each_trial_gets_a_fresh_cache(self):
        first = to_cache(self.tensors, self.model.config, "cpu")
        forward(self.model, self.suffix, "cpu", first)
        second = to_cache(self.tensors, self.model.config, "cpu")
        self.assertEqual((first.get_seq_length(), second.get_seq_length()), (8, 5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-cache-poc"))
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=[sys.argv[0]])
    sys.exit(run(args))
