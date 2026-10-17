# KV cache dataset investigation

Issue: [#21](https://github.com/jjustin-k/Tandem/issues/21). Updated: October 12, 2026.

## Recommendation

Use **FineWeb-Edu** for a small linear-mapping pilot. For a HeteroFold-style baseline, use question-bearing calibration prompts and collect receiver behavior targets as well as paired caches. Evaluate both approaches on separate documents and multi-turn tasks. Published KV artifacts exist, but I did not verify a ready-made dataset for two different-sized models that fits Tandem's MVP. This is a bounded search result, not a claim that no such dataset exists.

The [October 8 capstone notes](https://notes.granola.ai/d/6e8b1359-cf04-457a-883b-965684b3333d) propose translating caches between two small models, starting within one family. The data requirement starts with paired internal states for the same text prefix, rather than cache-transfer logs or mapper checkpoints. An output-aware translator additionally needs native receiver behavior targets. Model selection remains [issue #20](https://github.com/jjustin-k/Tandem/issues/20).

## Existing resources

| Resource | What is available | Use for Tandem |
| --- | --- | --- |
| [Llama FP16/BF16/INT8 artifacts](https://huggingface.co/datasets/taehyeonkim/fp16-bf16-int8) | Post-prefill tensors and prompt manifests for precision variants of Llama-3.1-8B-Instruct. The card specifies tensor keys, layout and metadata. | Useful serialization reference; not a different-model translation dataset. The INT8 snapshot omits the recent-token residual, so it is not a complete runtime cache. |
| [Cache-to-Cache (C2C)](https://github.com/thu-nics/C2C) | Dataset adapters, preparation scripts and training code; [released fusers](https://huggingface.co/nics-efc/C2C_Fuser/tree/main) are learned weights. | Reusable training-pipeline reference. The inspected code computes model caches during forward passes; its preparation scripts save text and generated responses, not a downloadable paired-tensor corpus. |
| [Cross-Model KV Cache Transfer in LLM Families](https://arxiv.org/html/2608.03893v1) | A calibration protocol: 500 FineWeb-Edu sequences of 1,024 tokens, sampled every fourth token for fitting. Uses matched KV head count/head dimension and factors positional rotation out of keys. | Closest precedent for creating paired calibration data. It reports uneven quality across model pairs; matching dimensions alone does not establish transfer quality. |
| [HeteroFold](https://arxiv.org/html/2609.32259v2) | Cross-family, receiver-prefill-free transfer. Calibration uses 1,600 training and 400 held-out prompts, balanced between Open-R1 problem text and HotpotQA training passages/questions, excluding answers and solutions. | Direct prior work. Its output-aware calibration needs native receiver queries, attention weights and post-output-projection attention outputs, beyond paired K/V. The [official repository](https://github.com/daniel-eai/Prefill-Free-Multi-Agent-LLMs) currently contains only a README announcing a forthcoming code release; no paired-data release was verified. |
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
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Linear baseline calibration prefixes | Public text with stable document IDs; directly used by the two mapping papers above. Stream a bounded sample rather than downloading the full corpus. |
| Open-R1 problem text + HotpotQA training passages/questions ([paper protocol](https://arxiv.org/html/2609.32259v2#A1.SS3)) | Output-aware calibration branch | Retain question spans, exclude answers/solutions, and pin the exact source dataset IDs/configurations before extraction. The paper does not fully specify reproducible source revisions or sampling IDs. |
| [SCBench](https://huggingface.co/datasets/microsoft/SCBench) | Held-out multi-turn/shared-context evaluation | Provides contexts, multiple questions and reference answers. Its `scbench_kv` task is textual key-value retrieval, not serialized transformer KV tensors. |
| [LongBench](https://huggingface.co/datasets/zai-org/LongBench) | Held-out QA, summarization and code evaluation | Tests whether mapped states preserve useful context across task types. Select examples that fit the chosen model's context window. |

FineWeb-Edu's card lists ODC-BY. Preserve dataset attribution and source identifiers. Before distributing generated artifacts, record the selected dataset and model licenses and their pinned revisions. Benchmark test data should not be used to fit or select the mapper.

## Generation protocol

The following is a proposed implementation, not an executed GPU experiment.

1. **Freeze the model pair and environment.** Select two dense, full-attention models in one family. Pin model/tokenizer revisions, PyTorch, Transformers, attention implementation and dtype. Qwen3-0.6B → Qwen3-1.7B is a small pilot candidate: both published configs have 28 layers, eight KV heads and head dimension 128 ([source](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json), [target](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json)). This does not replace issue #20's model decision or establish tokenizer compatibility.
2. **Build a prompt manifest first.** Choose the linear-prefix or question-bearing calibration branch above; the latter also records context/question character spans. Pin the corpus revision/configuration, stream order and seed. Select English documents long enough for one 1,024-token prefix. Save source document ID, text hash, split and exact token IDs. Deduplicate before splitting. Assign whole documents to train/validation/test; never randomly divide token rows from the same document among splits.
3. **Verify token alignment.** Use a fixed plain-text serialization for the pilot. Require both pinned tokenizers to produce identical IDs for every prefix, including special tokens. Do not independently apply different chat templates. If alignment fails, stop the simple paired-row pilot. A cross-family branch must save both token sequences, character-end offsets and an explicit correspondence table. HeteroFold aligns shared character boundaries and forward-fills unmatched receiver positions; special/template tokens require separate handling. Never pair unlike tokenizers by token index alone.
4. **Extract native caches.** Run each frozen model with `eval()`, `inference_mode()` and `use_cache=True`, batch size one. Capture post-prefill `past_key_values`; save per-layer keys and values as contiguous CPU tensors. Use two sequential model passes over the same manifest if simultaneous residency is too expensive. Record attention masks and positions. No answer generation is needed for cache extraction. For the output-aware branch, also capture receiver queries at question positions and the corresponding native attention/output targets. Extract pre-key-normalization/pre-RoPE keys through model-specific hooks when required; a runtime cache alone may not recover those features. Keep this capture separate from inference-time cache storage.
5. **Store bounded shards.** Use SafeTensors plus JSONL metadata, keyed by sample/model/layer. Store complete native caches for the first pilot. Record `[batch, kv_heads, tokens, head_dim]`, dtype, model revision/configuration, tokenizer revision, position IDs, RoPE settings, cache layout and extraction version. Keep raw post-RoPE keys; derive and label content-space keys only when implementing the mapper. A stride-four fitting view can reduce training data, but cannot be injected as a complete prefix cache.
6. **Fit and validate a baseline.** Start with a linear mapper and separate K/V losses; this pilot is not a HeteroFold reproduction. An output-aware branch must instead compare mapped attention and attention outputs with native receiver targets, using frozen receiver computation during offline calibration. Measure held-out reconstruction error, then target-model continuation quality. For a prefix of length T, map the first T−1 tokens and let the target consume the final token at its correct position to produce initial logits; compare with native target prefill of the same prefix. That final-token forward pass must be included in timing.
7. **Scale only after a pilot.** Proposed pilot: 32 training, eight validation and eight test documents. If extraction and cache replay work, expand the linear branch to 500/50/50 documents and test longer contexts and multi-turn handoffs. Size the output-aware branch separately after profiling; do not assume this count reproduces HeteroFold. Freeze hyperparameters on validation data; keep test results separate.

Use `uv` to manage the extraction environment and lock its dependencies. Generated caches belong in external artifact storage, with manifests and checksums in the repository; they should not be committed as large Git files.

### Output-aware capture schema (proposed)

Extend each sample manifest with receiver question positions, token correspondence, causal/padding masks, key-processing conventions and a behavior-target capture version. Store or reproducibly regenerate:

- native receiver queries at the selected question positions;
- native attention distributions over the accessible prefix;
- native attention outputs after the receiver output projection;
- receiver K/V and the model revision used to compute every target.

Receiver queries use query-head geometry, which can differ from KV-head geometry. Validate each tensor against the pinned model configuration. If querying a mapper outside the pilot's identical-tokenizer setting, form the receiver prefix/suffix split in receiver token coordinates and compare both arms on the same receiver suffix.

Avoid storing full attention matrices for every token and layer: their size grows quadratically with context length. Profile question-span capture first, then choose between bounded probe shards and recomputing frozen receiver targets during calibration. A reduced probe budget is a Tandem experiment, not an exact reproduction of the paper. Record the chosen policy in the manifest. Inference-mode extraction produces fixed targets; gradient-based mapper calibration must still allow gradients through mapped receiver attention.

## Cost and correctness gates

For dense attention, uncompressed cache payload per model is:

`2 × layers × KV heads × head dimension × tokens × bytes per element`

For the proposed Qwen pair at BF16 and 1,024 tokens, this gives **112 MiB per model, 224 MiB per pair**. A 48-document pilot is approximately **10.5 GiB**; 500 training pairs alone are **109.4 GiB**, or **131.3 GiB** including 50 validation and 50 test pairs. These calculated payloads cover **paired runtime K/V only**. They exclude receiver queries, attention distributions/outputs, weights, temporary activations, mapper tensors and metadata. An output-aware dataset needs a separate measured storage budget. Stride-four storage would be smaller, but loses the full runtime cache needed for replay checks.

Measure ten pilot extractions to estimate wall time and peak GPU/CPU memory before requesting a larger run. Record extraction throughput and disk-write time separately. No generation-time or GPU-fit claim has been validated here.

Before training, require matching raw-text hashes, verified identical IDs or a validated correspondence table, expected tensor shapes, finite values, complete layer counts and serialization round-trip checks. For behavior targets, verify question spans and causal masks, normalize attention over valid positions, and compare regenerated targets with their stored versions within the fixed tolerance. Replay a native target cache through the same injection path and compare its next-token logits with native prefill within a tolerance fixed for the selected dtype/backend. This isolates cache-position or injection bugs from mapper error.

Compare mapped handoff against target native prefill, source-only generation and a deliberately invalid/direct-copy baseline where shapes permit. Report downstream accuracy or answer quality, perplexity/logit divergence, handoff latency, bytes moved and peak memory. A cache hit only shows that state was reused; it does not show that the translated state is correct. Distinguish warm handoff (source cache already exists) from cold execution (source prefill must also be paid).

## Issue outcome

There is enough public prompt data and a published calibration protocol to generate the required supervision. The next concrete step is choosing the model pair in #20 and running the bounded extraction pilot above. Neither paired tensors nor mapper quality were produced in this research task. The cross-model mapping and CacheBridge papers cover within-family transfer, while HeteroFold directly addresses receiver-prefill-free cross-family transfer. Both are prior work. Reliable semantic-router-driven handoffs could be a direction to investigate, but neither their novelty nor their quality is established by this report.
