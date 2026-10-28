# Paired prefill feature dataset

Issue: [#45](https://github.com/tandemkv/Tandem/issues/45).

Generate paired features for the first cross-model mapper. Both models are
frozen. The script loads one model at a time and generates no answers.

## Run the ten-prompt pilot

From the repository root:

```sh
uv run --script test/experiments/test_paired_kv_dataset.py
uv run --script src/experiments/paired_kv_dataset.py prepare
uv run --script src/experiments/paired_kv_dataset.py extract
uv run --script src/experiments/paired_kv_dataset.py verify
```

Python 3.12 dependencies and model/data revisions are pinned. The default
device is CUDA if available, otherwise MPS, otherwise CPU. Use `--device cpu`,
`--device mps` or `--device cuda` on `extract` to choose explicitly.

Preparation downloads a pinned HotpotQA training Parquet shard (about 158 MiB).
Qwen weights are about 1 GB and SmolLM weights about 693 MiB; the Hub cache
reuses existing downloads. Inference and stored features use FP32. This is a
correctness/extraction pilot, not an optimized extraction benchmark.

## Models and source data

| Alias | Model | Pinned revision |
| --- | --- | --- |
| qwen | Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |
| smol | HuggingFaceTB/SmolLM2-360M-Instruct | `a10cc1512eabd3dde888204e902eca88bddb4951` |

The source is [HotpotQA](https://hotpotqa.github.io/), available on the Hub as
`hotpotqa/hotpot_qa`, revision `1908d6afbbead072334abe2965f91bd2709910ab`,
file `distractor/train-00000-of-00002.parquet`. Its dataset card lists
**CC-BY-SA-4.0**. Preserve attribution and the applicable license when sharing
the text or derived dataset. Both model cards list Apache-2.0.

This uses one of [HeteroFold's](https://arxiv.org/abs/2609.32259) text sources,
not its exact calibration manifest. The pilot excludes Open-R1 and selects
short complete HotpotQA examples. It cannot be reported as a reproduction of
the paper's mixed-corpus results.

## Preparation and splits

The sampler reads only ID, question and context columns; answer annotations
are excluded. It shuffles the first 4,096 source rows with seed `20261028`,
then selects ten complete contexts that fit within 1,024 tokens under **both**
tokenizers. It never truncates passages or questions. The last two selected
examples are validation; the other eight are training. Normalized document
titles cannot occur in more than one selected example, so these splits share
no titles. Text hashes and sample IDs are unique.

The text format is the complete distractor passages followed by `Question:`
and an empty `Answer:` cue. No chat templates or special tokens are inserted.
Both models see identical raw text, each tokenized independently. The manifest
records each model's exact token IDs, character offsets and question-token
positions, plus source row/ID, raw text hash and split. Use validation to select
mapper settings; add a separate, document-disjoint test set for final quality
evaluation. These ten examples only validate the pipeline.

## Features and alignment

Hooks capture every decoder layer's **k_proj output before RoPE** and
**v_proj output**, including projection biases. No key normalization is present
in the two supported architectures. Each tensor is `[tokens, KV_heads × head_dim]`,
on CPU in FP32. These are training features, not directly injectable runtime
caches. A future mapper must apply the destination model's positional rotation
exactly once when building its runtime cache.

Alignment follows HeteroFold's shared character-end boundary approach:

- `exact_pairs` gives matched source/target token indices for paired fitting.
- `target_to_source` uses the latest preceding shared boundary at unmatched
  positions. It is recorded separately for each transfer direction.
- Positions before the first shared boundary use the first source token and
  are explicitly marked `before_first`. This fallback is not claimed to have
  an exact matching causal endpoint. At least eight shared boundaries are required.

This pilot captures K/V only. HeteroFold-style output-aware calibration also
needs native receiver queries/attention/output targets. The saved question
positions and pinned inputs allow those to be regenerated later; they are not
silently represented as part of this dataset.

## Output and checks

All generated files stay under the Git-ignored `artifacts/paired-kv-pilot/`:

```text
manifest.json                  text, provenance, splits, tokens and alignment
qwen/<sample-id>.safetensors    per-layer K/V features
qwen/<sample-id>.json           geometry, identity, SHA-256 and extraction time
smol/<sample-id>.safetensors
smol/<sample-id>.json
validation.json                result of checking the complete dataset
```

Each file has embedded model/text/manifest identity. Extraction checks complete
layer counts, shapes, dtype, finite values and exact serialization round trips.
Verification rechecks provenance, file hashes, shapes, alignment and document
separation, and fails if either model's shards are absent. Model revisions and
feature-space version must match the manifest.

Preparation and extraction refuse to overwrite existing manifests or shards.
To recover a partial extraction, retain completed model outputs and run
`extract --models smol` or `extract --models qwen` for the missing model. For a
partially completed model directory, use a new output directory or explicitly
move the incomplete model directory aside before rerunning it. No automatic
deletion or unverified resume occurs.

## Scale on school CUDA compute

```sh
uv run --script src/experiments/paired_kv_dataset.py prepare \
  --output artifacts/paired-kv-200 --count 200 --scan-rows 20000
uv run --script src/experiments/paired_kv_dataset.py extract \
  --output artifacts/paired-kv-200 --device cuda
uv run --script src/experiments/paired_kv_dataset.py verify \
  --output artifacts/paired-kv-200
```

This gives 160 training / 40 validation examples when enough eligible contexts
are found. It is still a HotpotQA-only variant. Allow up to roughly 20.3 GiB
for 200 paired FP32 examples at 1,024 tokens per model, excluding source data,
weights and temporary memory. Actual payload is lower for shorter examples.
The CUDA path must be measured on the school allocation; local MPS results do
not validate CUDA performance or available cluster capacity.

## Pilot result, October 28, 2026

The local M2 run produced **20 validated shards, 775.6 MiB**, from ten prompts
(eight training / two validation). Qwen processed 7,234 tokens across 24 layers
with feature width 128; Smol processed 7,756 tokens across 32 layers with width
320. Complete contexts fit within 1,024 tokens under both tokenizers.

Captured-feature extraction took 12.81 seconds total for Qwen and 18.12 seconds
for Smol. These times include hook copies to CPU and exclude model loading,
downloads, serialization and final verification. They are not TTFT measurements.
Both models ran sequentially on MPS in FP32.

All tensors survived serialization exactly. Final verification checked 20 file
hashes, embedded provenance, the pinned model geometries, finite values, text
hashes, question spans, alignment and document separation. Six offline tests
passed, including reconstruction of native runtime keys by applying RoPE to
the captured key projections for both supported architecture classes.

The [verification record](../../refs/experiments/paired-kv-pilot-2026-10-28.json)
contains source/model revisions, sampler settings, sample IDs/splits, per-shard
hashes, dimensions and alignment counts. The raw manifest and tensors remain
in the local ignored artifact directory. No mapper has been fitted and no
output-aware attention targets or final test set have been collected.
