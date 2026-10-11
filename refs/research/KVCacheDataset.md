# KV cache dataset investigation

Issue: [#21](https://github.com/jjustin-k/Tandem/issues/21). Checked: October 11, 2026.

## Recommendation

Generate paired source/target caches from a small, reproducible sample of **FineWeb-Edu**, then evaluate on separate documents and multi-turn tasks. Published KV artifacts exist, but I did not verify a ready-made dataset for two different-sized models that fits Tandem's MVP. This is a bounded search result, not a claim that no such dataset exists.

The [October 8 capstone notes](https://notes.granola.ai/d/6e8b1359-cf04-457a-883b-965684b3333d) propose translating caches between two small models, starting within one family. The data requirement is therefore pairs of internal states for the same tokenized prefix, rather than cache-transfer logs, prompt/answer pairs, or mapper checkpoints. Model selection remains [issue #20](https://github.com/jjustin-k/Tandem/issues/20).

## Existing resources

| Resource | What is available | Use for Tandem |
| --- | --- | --- |
| [Llama FP16/BF16/INT8 artifacts](https://huggingface.co/datasets/taehyeonkim/fp16-bf16-int8) | Post-prefill tensors and prompt manifests for precision variants of Llama-3.1-8B-Instruct. The card specifies tensor keys, layout and metadata. | Useful serialization reference; not a different-model translation dataset. The INT8 snapshot omits the recent-token residual, so it is not a complete runtime cache. |
| [Cache-to-Cache (C2C)](https://github.com/thu-nics/C2C) | Dataset adapters, preparation scripts and training code; [released fusers](https://huggingface.co/nics-efc/C2C_Fuser/tree/main) are learned weights. | Reusable training-pipeline reference. The inspected code computes model caches during forward passes; its preparation scripts save text and generated responses, not a downloadable paired-tensor corpus. |
| [Cross-Model KV Cache Transfer in LLM Families](https://arxiv.org/html/2608.03893v1) | A calibration protocol: 500 FineWeb-Edu sequences of 1,024 tokens, sampled every fourth token for fitting. Uses matched KV head count/head dimension and factors positional rotation out of keys. | Closest precedent for creating paired calibration data. It reports uneven quality across model pairs; matching dimensions alone does not establish transfer quality. |
| [CacheBridge](https://arxiv.org/html/2609.00891v1) | Also evaluates with 500 FineWeb-Edu sequences of length 1,024 and stride-four sampling. | Relevant alternative mapping baseline, not a verified downloadable paired-cache dataset. |
| [EchoKV](https://arxiv.org/html/2603.22910v1) | Same-model cache reconstruction trained with LongAlpaca; [official code](https://github.com/noforit/EchoKV) includes a NarrativeQA demonstration. | Reference for the compression fallback. Its partial/full-cache supervision is a different task from cross-model translation. |

The C2C code inspection used commit `bdb932aa63637959074a9a6fd59992a4d0d9ad22`: [dataset adapters](https://github.com/thu-nics/C2C/blob/bdb932aa63637959074a9a6fd59992a4d0d9ad22/rosetta/train/dataset_adapters.py), [text preparation](https://github.com/thu-nics/C2C/blob/bdb932aa63637959074a9a6fd59992a4d0d9ad22/script/dataset/create_openhermes.py), and [model wrapper](https://github.com/thu-nics/C2C/blob/bdb932aa63637959074a9a6fd59992a4d0d9ad22/rosetta/model/wrapper.py). C2C projects and fuses caches, including receiver computation; it must not be treated as proof of a complete target-prefill replacement.

### Other Hub results checked

Hub dataset searches for `kv-cache` and `kv cache`, plus paper/repository searches, found:

- [st192011/KVCaches](https://huggingface.co/datasets/st192011/KVCaches): three `.bin` files and a license-only card. Model identity and cross-model pairing were not documented in the inspected metadata.
- [Visionarys/kvcache](https://huggingface.co/datasets/Visionarys/kvcache): one ZIP archive with no dataset card. Contents and suitability remain unverified.
- [ergt2025/kvcache_offload](https://huggingface.co/datasets/ergt2025/kvcache_offload): profiling logs; no paired cache corpus established.
- [jprivera44/kv_cache_logits_exp2](https://huggingface.co/datasets/jprivera44/kv_cache_logits_exp2): an evaluation JSON; no paired cache corpus established.

Cards/file listings were inspected without downloading large tensor archives. The documented Llama artifacts were checked at revision `e70889d57e074bd727aae0c6f10803ecc4934694`; tensors were not loaded or independently validated. Undocumented archives are unresolved candidates, not evidence of absence.

## Prompt data

| Dataset | Proposed role | Reason |
| --- | --- | --- |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Initial calibration/training prefixes | Public text with stable document IDs; directly used by the two mapping papers above. Stream a bounded sample rather than downloading the full corpus. |
| [SCBench](https://huggingface.co/datasets/microsoft/SCBench) | Held-out multi-turn/shared-context evaluation | Provides contexts, multiple questions and reference answers. Its `scbench_kv` task is textual key-value retrieval, not serialized transformer KV tensors. |
| [LongBench](https://huggingface.co/datasets/zai-org/LongBench) | Held-out QA, summarization and code evaluation | Tests whether mapped states preserve useful context across task types. Select examples that fit the chosen model's context window. |

FineWeb-Edu's card lists ODC-BY. Preserve dataset attribution and source identifiers. Before distributing generated artifacts, record the selected dataset and model licenses and their pinned revisions. Benchmark test data should not be used to fit or select the mapper.

## Generation protocol

The following is a proposed implementation, not an executed GPU experiment.

1. **Freeze the model pair and environment.** Select two dense, full-attention models in one family. Pin model/tokenizer revisions, PyTorch, Transformers, attention implementation and dtype. Qwen3-0.6B → Qwen3-1.7B is a small pilot candidate: both published configs have 28 layers, eight KV heads and head dimension 128 ([source](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json), [target](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json)). This does not replace issue #20's model decision or establish tokenizer compatibility.
2. **Build a prompt manifest first.** Pin the corpus revision/configuration, stream order and seed. Select English documents long enough for one 1,024-token prefix. Save source document ID, text hash, split and exact token IDs. Deduplicate before splitting. Assign whole documents to train/validation/test; never randomly divide token rows from the same document among splits.
3. **Verify token alignment.** Use a fixed plain-text serialization for the pilot. Require both pinned tokenizers to produce identical IDs for every prefix, including special tokens. Do not independently apply different chat templates. If alignment fails, stop: unequal token sequences require an explicit alignment method, not simple row pairing.
4. **Extract native caches.** Run each frozen model with `eval()`, `inference_mode()` and `use_cache=True`, batch size one. Capture post-prefill `past_key_values`; save per-layer keys and values as contiguous CPU tensors. Use two sequential model passes over the same manifest if simultaneous residency is too expensive. Record attention masks and positions. No answer generation is needed for this calibration step.
5. **Store bounded shards.** Use SafeTensors plus JSONL metadata, keyed by sample/model/layer. Store complete native caches for the first pilot. Record `[batch, kv_heads, tokens, head_dim]`, dtype, model revision/configuration, tokenizer revision, position IDs, RoPE settings, cache layout and extraction version. Keep raw post-RoPE keys; derive and label content-space keys only when implementing the mapper. A stride-four fitting view can reduce training data, but cannot be injected as a complete prefix cache.
6. **Fit and validate a baseline.** Start with a linear mapper and separate K/V losses. Measure held-out reconstruction error, then target-model continuation quality. For a prefix of length T, map the first T−1 tokens and let the target consume the final token at its correct position to produce initial logits; compare with native target prefill of the same prefix. That final-token forward pass must be included in timing.
7. **Scale only after a pilot.** Proposed pilot: 32 training, eight validation and eight test documents. If extraction and cache replay work, expand to 500/50/50 documents and test longer contexts and multi-turn handoffs. Freeze hyperparameters on validation data; keep test results separate.

Use `uv` to manage the extraction environment and lock its dependencies. Generated caches belong in external artifact storage, with manifests and checksums in the repository; they should not be committed as large Git files.

## Cost and correctness gates

For dense attention, uncompressed cache payload per model is:

`2 × layers × KV heads × head dimension × tokens × bytes per element`

For the proposed Qwen pair at BF16 and 1,024 tokens, this gives **112 MiB per model, 224 MiB per pair**. A 48-document pilot is approximately **10.5 GiB**; 500 training pairs alone are **109.4 GiB**, or **131.3 GiB** including 50 validation and 50 test pairs. These calculated payloads exclude weights, temporary activations, mapper tensors and metadata. Stride-four storage would be smaller, but loses the full runtime cache needed for replay checks.

Measure ten pilot extractions to estimate wall time and peak GPU/CPU memory before requesting a larger run. Record extraction throughput and disk-write time separately. No generation-time or GPU-fit claim has been validated here.

Before training, require matching prompt hashes/token IDs, expected tensor shapes, finite values, complete layer counts and serialization round-trip checks. Replay a native target cache through the same injection path and compare its next-token logits with native prefill within a tolerance fixed for the selected dtype/backend. This isolates cache-position or injection bugs from mapper error.

Compare mapped handoff against target native prefill, source-only generation and a deliberately invalid/direct-copy baseline where shapes permit. Report downstream accuracy or answer quality, perplexity/logit divergence, handoff latency, bytes moved and peak memory. A cache hit only shows that state was reused; it does not show that the translated state is correct. Distinguish warm handoff (source cache already exists) from cold execution (source prefill must also be paid).

## Issue outcome

There is enough public prompt data and a published calibration protocol to generate the required supervision. The next concrete step is choosing the model pair in #20 and running the bounded extraction pilot above. Neither paired tensors nor mapper quality were produced in this research task. The cross-model mapping and CacheBridge papers also mean that basic within-family transfer should be evaluated as prior work; novelty requires a narrower, verified contribution.
