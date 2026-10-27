# Local KV cache proof of concept

Related: [issue #32](https://github.com/tandemkv/Tandem/issues/32) and
[Justin's initial extraction PR #33](https://github.com/tandemkv/Tandem/pull/33).

## Question

Can a small model on a 16 GB Apple Silicon Mac continue from its own saved
context with the same outputs as fresh inference, and what does reuse cost?

This is the first control for Tandem's cross-model experiment. If native cache
replay is incorrect, a failed learned transfer cannot be attributed to the
mapper. It does not establish cross-model transfer, memory savings, or novel
research.

## Run

From the repository root, with `uv` installed:

```sh
uv run --script test/experiments/test_local_cache_poc.py
uv run --script src/experiments/local_cache_poc.py
```

The second command downloads the pinned Qwen2.5-0.5B-Instruct checkpoint
(roughly 1 GB of weights), uses MPS when available, and writes
`artifacts/local-cache-poc/results.json`. Dependencies are pinned in the script;
its adjacent lockfile pins their resolution. Model weights use revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Inference is FP32 with eager attention
to make correctness comparisons easier to interpret. No model training occurs.

For a quick installation check:

```sh
uv run --script src/experiments/local_cache_poc.py --lengths 128 --repeats 1
```

Use `--device cpu` to test the CPU path, and a different `--output` directory
to retain separate runs. The experiment keeps only one serialized cache at a
time and deletes it after that case. It does not download HeteroFold's models
or calibration dataset. Artifacts are ignored by Git.

## Controlled comparison

Nine synthetic cases put a random four-character project code near the start,
middle, or end of prefixes of 128, 512, and 1,024 tokens. Filler carries no
release codes. The same model answers the same suffix question in each arm:

| Arm | Processing measured |
| --- | --- |
| Native | Full prefix plus new question, through selection of the first answer token |
| Warm | New question with an already prepared cache on the inference device |
| Restored | Read a SafeTensors cache from disk, validate identity/shape/dtype/finiteness, move it to the device, and process the question |
| No context | Answer the question without the project notes; quality control only |

All three cache arms use **identical token IDs**, attention masks, and positions.
No chat-template differences are hidden inside the comparison. Every timed
trial gets a fresh cache instance because decoding mutates the cache.

Device operations are synchronized around timing. The model is warmed up,
and the three timed arms run in seeded shuffled order for three repetitions
per case. Each arm also generates up to 16 greedy tokens once. The script
records every duration and answer rather than only averages. Prefix creation,
serialization, model loading, and initial downloads are outside warm timing;
prefix creation and serialization are recorded separately. Restored timing
includes validation and device copies. Disk reads can benefit from the OS
file cache: this is not a cold-storage benchmark.

The output includes first-layer shape, all-layer payload bytes, file size,
process RSS, system available memory, and sampled MPS allocation/driver memory.
RSS and MPS memory can overlap on unified memory; **do not add them**. Sampled
MPS maxima are observations, not guaranteed instantaneous peaks. The process
RSS high-water mark is also recorded. These are single-request measurements,
not serving throughput or production GPU measurements.

## Gates before translating caches

1. Serialization preserves every tensor exactly. Wrong identity, missing
   layers, truncated shapes, wrong dtype, and non-finite values are rejected.
2. Warm and restored next-token logits match fresh inference with fixed
   `atol=rtol=0.001`; generated token IDs match exactly. No tolerance is tuned
   after examining a failed run.
3. Native inference answers at least 8/9 planted facts correctly, while the
   no-context control answers none. If native QA fails, revise the task using
   separate development examples before evaluating a mapper.
4. Report measured memory and all three latency distributions, including
   cases where loading a cache is slower. A speedup is not a correctness gate.

The small fixed case set is a diagnostic, not a statistically powered quality
evaluation. Repeating the same prompts improves timing evidence, not the
number of independent quality examples.

## Next experiment, if these gates pass

Use Qwen3-0.6B to Qwen3-1.7B as a candidate, subject to actual tokenizer checks.
Extract the two models sequentially so their weights need not coexist in
memory. Begin with short contexts and a bounded document-level train/validation
split. Before fitting anything, verify native cache replay for the new model
pair's target through exactly the injection path the mapper will use.

Measure a mapped cache against native target inference and a no-context
control on unseen facts and documents. Add an untrained/direct-copy control
when tensor geometry permits. Judge task quality and downstream logits as
well as cache error. Key normalization and RoPE must be handled exactly once.
Only the mapper is fitted; base model weights stay frozen. Keep this a separate
experiment so native replay success cannot be mistaken for translation success.

[HeteroFold](https://arxiv.org/html/2609.32259v2) motivates evaluating receiver
behavior rather than only cache reconstruction. Its
[official implementation](https://github.com/daniel-eai/Prefill-Free-Multi-Agent-LLMs)
now exists, but uses CUDA and much larger models. Adapting its full calibration
pipeline to this Mac is separate work. The first local control needs neither
its training pipeline nor a semantic router.

## Measured on the M2 Air, October 27, 2026

The default run completed on a 16 GB M2 MacBook Air using MPS, FP32, PyTorch
2.9.1, and Transformers 4.57.1. The
[full result record](../../refs/experiments/local-cache-m2-2026-10-27.json)
includes all cases, timing samples, tensor sizes, and memory observations.

| Prefix tokens | Native first token | Warm first token | Restored first token | Cache payload |
| --- | ---: | ---: | ---: | ---: |
| 128 | 82.55 ms | 43.03 ms | 63.08 ms | 3 MiB |
| 512 | 695.70 ms | 68.19 ms | 95.23 ms | 12 MiB |
| 1,024 | 1,642.10 ms | 103.57 ms | 131.86 ms | 24 MiB |

Each cell is the median of nine measurements: three fact positions, with
three timing repetitions each. These observed reductions assume that a
valid same-model prefix cache already exists. They are not expected mapper
speedups, and cache preparation/serialization costs have not disappeared.

- Both replay arms matched all nine native greedy continuations exactly.
  Maximum next-token logit error was `5.43594e-5`, within the fixed tolerance.
- Native and both replay arms retrieved 8/9 codes. All three answered
  `UNKNOWN` instead of `EQBU` in the 128-token late-fact case. No-context
  inference retrieved 0/9. The model's one failure is retained in the results.
- Process peak RSS was 2.98 GiB; maximum sampled MPS driver allocation was
  2.73 GiB. These overlap and must not be summed. Minimum sampled system
  available memory was 2.39 GiB with other applications running.
- Four offline tests passed, including native-versus-restored replay and
  rejection of wrong identity, truncation, missing layers, non-finite values,
  and wrong dtype. Static checks passed.

This establishes that **this bounded control runs on this Mac**. It does not
establish that two larger models or HeteroFold calibration fit. The next
decision is whether a fitted cross-model mapper preserves enough behavior on
unseen data; the replay result alone cannot answer that.
