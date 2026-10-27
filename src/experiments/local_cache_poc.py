# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.9.1", "transformers==4.57.1", "safetensors==0.6.2",
#   "numpy==2.3.4", "psutil==7.1.2",
# ]
# ///
"""Bounded same-model KV replay control; not a cross-model transfer result."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import resource
import statistics
import sys
import time
from pathlib import Path

import psutil
import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
ATOL = RTOL = 1e-3


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def memory(device):
    result = {"rss_bytes": psutil.Process().memory_info().rss,
              "system_available_bytes": psutil.virtual_memory().available}
    if device == "mps":
        result.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())
    return result


def tensors_from_cache(cache):
    return {f"{i}.{role}": tensor.detach().cpu().contiguous()
            for i, layer in enumerate(cache.layers)
            for role, tensor in (("k", layer.keys), ("v", layer.values))}


def validate_tensors(tensors, config, length, dtype):
    expected = {f"{i}.{role}" for i in range(config.num_hidden_layers)
                for role in ("k", "v")}
    if set(tensors) != expected:
        raise ValueError("Missing or unexpected cache layers/roles")
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads)
    shape = (1, config.num_key_value_heads, length, head_dim)
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"Invalid cache shape/dtype: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite cache: {name}")


def cache_from_tensors(tensors, config, device):
    cache = DynamicCache(config=config)
    for i in range(config.num_hidden_layers):
        # Clone even on CPU: inference extends caches in place.
        cache.update(tensors[f"{i}.k"].to(device).clone(),
                     tensors[f"{i}.v"].to(device).clone(), i)
    return cache


def save_cache(path, tensors, identity):
    save_file(tensors, str(path), metadata={"identity": json.dumps(identity, sort_keys=True)})


def restore_cache(path, identity, config, device, dtype):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as stored:
        actual = json.loads(stored.metadata()["identity"])
    if actual != identity:
        raise ValueError("Cache identity does not match model, prefix or runtime")
    tensors = load_file(str(path), device="cpu")
    validate_tensors(tensors, config, identity["prefix_length"], dtype)
    return cache_from_tensors(tensors, config, device)


def identity_for(prefix):
    return {"model": MODEL, "revision": REVISION, "prefix_length": len(prefix),
            "token_ids_sha256": digest(prefix), "dtype": "float32",
            "transformers": transformers.__version__, "format_version": 1}


def make_case(tokenizer, length, position, seed):
    rng = random.Random(seed)
    code = "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=4))
    name = "Cedar"
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    header = encode("<|im_start|>system\nAnswer using only the provided project notes. "
                    "If the answer is absent, say UNKNOWN.<|im_end|>\n<|im_start|>user\n")
    fact = encode(f"\nProject {name}'s release code is {code}.\n")
    filler = encode("The archive contains routine notes about schedules, meetings, "
                    "testing, and documentation. These notes contain no release codes.\n")
    available = length - len(header) - len(fact)
    if available < 0:
        raise ValueError("Context too short for the fact and template")
    before = {"early": 0, "middle": available // 2, "late": available}[position]
    repeated = (filler * (length // len(filler) + 2))
    prefix = header + repeated[:before] + fact + repeated[:available - before]
    suffix = encode(f"\nWhat is Project {name}'s release code? "
                    "Answer with the code only.<|im_end|>\n<|im_start|>assistant\n")
    return {"prefix": prefix, "suffix": suffix, "no_context": header + suffix,
            "expected": code, "position": position, "seed": seed,
            "prefix_text": tokenizer.decode(prefix)}


@torch.inference_mode()
def forward(model, tokens, device, cache=None):
    past = cache.get_seq_length() if cache is not None else 0
    count = len(tokens)
    return model(input_ids=torch.tensor([tokens], device=device),
                 attention_mask=torch.ones((1, past + count), dtype=torch.long, device=device),
                 position_ids=torch.arange(past, past + count, device=device)[None, :],
                 cache_position=torch.arange(past, past + count, device=device),
                 past_key_values=cache, use_cache=True, logits_to_keep=1)


@torch.inference_mode()
def complete(model, first, device, limit, eos):
    output = first
    tokens = []
    for step in range(limit):
        token = int(output.logits[0, -1].argmax())
        tokens.append(token)
        if token in eos or step == limit - 1:
            break
        output = forward(model, [token], device, output.past_key_values)
    return tokens


def run(args):
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; choose --device cpu explicitly")
    if args.repeats < 1 or args.max_new_tokens < 1:
        raise ValueError("Repeat and generation limits must be positive")
    if any(n < 96 or n > 2048 for n in args.lengths):
        raise ValueError("This local POC supports prefix lengths from 96 to 2048")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    print(f"Loading {MODEL} at {REVISION[:8]} on {device} (float32)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        attn_implementation="eager").eval().to(device)
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    report = {"model": MODEL, "revision": REVISION, "device": device,
              "dtype": "float32", "attention": "eager", "seed": args.seed,
              "python": platform.python_version(), "torch": torch.__version__,
              "transformers": transformers.__version__, "platform": platform.platform(),
              "system_memory_bytes": psutil.virtual_memory().total,
              "atol": ATOL, "rtol": RTOL, "repeats": args.repeats,
              "max_new_tokens": args.max_new_tokens, "memory_samples": [], "cases": []}
    report["memory_samples"].append(memory(device))
    # Warm up the device/model before measuring. Compilation/loading is excluded.
    warmup = forward(model, [tokenizer.eos_token_id], device)
    del warmup
    sync(device)
    for length in args.lengths:
        for case_index, position in enumerate(("early", "middle", "late")):
            case = make_case(tokenizer, length, position, args.seed + case_index + length)
            prefix, suffix = case["prefix"], case["suffix"]
            ident = identity_for(prefix)
            sync(device)
            start = time.perf_counter()
            prefetched = forward(model, prefix, device)
            sync(device)
            preparation_ms = (time.perf_counter() - start) * 1000
            tensors = tensors_from_cache(prefetched.past_key_values)
            del prefetched
            validate_tensors(tensors, model.config, length, torch.float32)
            path = args.output / f"cache-{length}-{position}.safetensors"
            start = time.perf_counter()
            save_cache(path, tensors, ident)
            serialization_ms = (time.perf_counter() - start) * 1000
            restored = load_file(str(path), device="cpu")
            if not all(torch.equal(restored[k], v) for k, v in tensors.items()):
                raise AssertionError("Cache serialization changed tensor values")
            del restored
            result = {"prefix_length": length, "suffix_length": len(suffix),
                      "fact_position": position, "expected": case["expected"],
                      "prefix_sha256": ident["token_ids_sha256"],
                      "prefix_text": case["prefix_text"], "seed": case["seed"],
                      "prefix_preparation_ms": preparation_ms,
                      "serialization_ms": serialization_ms,
                      "cache_payload_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
                      "cache_file_bytes": path.stat().st_size,
                      "first_layer_shape": list(tensors["0.k"].shape),
                      "layers": model.config.num_hidden_layers, "arms": {}}
            timings = {arm: [] for arm in ("native", "warm", "restored")}
            first_logits, generated = {}, {}
            rng = random.Random(case["seed"])
            for repetition in range(args.repeats):
                order = list(timings)
                rng.shuffle(order)
                for arm in order:
                    # Prepare the in-memory cache outside warm timing, on device.
                    cache = cache_from_tensors(tensors, model.config, device) if arm == "warm" else None
                    sync(device)
                    start = time.perf_counter()
                    if arm == "restored":
                        cache = restore_cache(path, ident, model.config, device, torch.float32)
                    output = forward(model, prefix + suffix if arm == "native" else suffix, device, cache)
                    int(output.logits[0, -1].argmax())  # Include first-token selection/readback.
                    sync(device)
                    timings[arm].append((time.perf_counter() - start) * 1000)
                    if output.past_key_values.get_seq_length() != length + len(suffix):
                        raise AssertionError("Cache length does not cover the exact prompt")
                    if repetition == 0:
                        first_logits[arm] = output.logits[0, -1].float().cpu()
                        decode_start = time.perf_counter()
                        generated[arm] = complete(model, output, device, args.max_new_tokens, eos)
                        sync(device)
                        decode_ms = (time.perf_counter() - decode_start) * 1000
                        final_length = output.past_key_values.get_seq_length()
                        if final_length != length + len(suffix) + len(generated[arm]) - 1:
                            raise AssertionError("Decode appended an unexpected number of cache positions")
                        result["arms"][arm] = {"generated_ids": generated[arm],
                            "answer": tokenizer.decode(generated[arm], skip_special_tokens=True).strip(),
                            "continuation_ms": decode_ms, "final_cache_length": final_length}
                    report["memory_samples"].append(memory(device))
                    del output, cache
            for arm, values in timings.items():
                entry = result["arms"][arm]
                entry.update(ttft_ms=values, median_ttft_ms=statistics.median(values),
                             exact_answer=entry["answer"] == case["expected"])
                if arm != "native":
                    reference, actual = first_logits["native"], first_logits[arm]
                    entry["max_abs_logit_error"] = float((reference - actual).abs().max())
                    entry["logits_match"] = bool(torch.isfinite(actual).all() and
                                                torch.allclose(reference, actual, atol=ATOL, rtol=RTOL))
                    entry["greedy_tokens_match"] = generated[arm] == generated["native"]
            empty = forward(model, case["no_context"], device)
            empty_tokens = complete(model, empty, device, args.max_new_tokens, eos)
            del empty
            empty_answer = tokenizer.decode(empty_tokens, skip_special_tokens=True).strip()
            result["no_context"] = {"answer": empty_answer,
                                    "exact_answer": empty_answer == case["expected"]}
            report["cases"].append(result)
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{length}/{position}: native={result['arms']['native']['answer']!r}; "
                  f"warm={result['arms']['warm']['answer']!r}; "
                  f"restore={result['arms']['restored']['answer']!r}", flush=True)
            path.unlink()  # Keep one cache on disk at a time; result records its size.
            del tensors, first_logits, generated
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
    replay_ok = all(c["arms"][a]["logits_match"] and c["arms"][a]["greedy_tokens_match"]
                    for c in report["cases"] for a in ("warm", "restored"))
    report["replay_passed"] = replay_ok
    report["native_exact_answers"] = sum(c["arms"]["native"]["exact_answer"] for c in report["cases"])
    report["no_context_exact_answers"] = sum(c["no_context"]["exact_answer"] for c in report["cases"])
    report["fact_retrieval_passed"] = (
        report["native_exact_answers"] >= math.ceil(8 * len(report["cases"]) / 9)
        and report["no_context_exact_answers"] == 0)
    report["poc_passed"] = replay_ok and report["fact_retrieval_passed"]
    report["observed_max_rss_bytes"] = max(m["rss_bytes"] for m in report["memory_samples"])
    report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024)
    if device == "mps":
        report["observed_max_mps_driver_bytes"] = max(m["mps_driver_bytes"] for m in report["memory_samples"])
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in (
        "poc_passed", "replay_passed", "fact_retrieval_passed", "native_exact_answers", "no_context_exact_answers",
        "observed_max_rss_bytes", "observed_max_mps_driver_bytes")}), flush=True)
    return 0 if report["poc_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-cache-poc"))
    sys.exit(run(parser.parse_args()))
