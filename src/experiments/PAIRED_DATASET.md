# Paired KV dataset

The [private Hugging Face dataset](https://huggingface.co/datasets/tandemkv/paired-kv-qwen-smollm) contains the validated 200-prompt HotpotQA run, labels and model features. Its card describes the data, licensing and split. The feature manifest is immutable because each shard records its hash; `labels.json` joins answers by sample ID.

`paired_kv_dataset.py` prepares complete prompts, token boundaries and splits; extracts frozen model key projections before RoPE and value projections; and verifies local shards. `cloud_prefill.py` uses that extractor to upload and verify batches before deleting temporary copies. `replay_batch.py` downloads one shard at a time and runs `replay_cache.py` to compare saved-cache next-token logits against native prefill. `labels` checks the original source rows and text before joining answers.

The scripts pin the source and model revisions. Use Python through `uv`; do not put model weights, tokens, output features, reports or credentials in Git. Log in with `uvx hf auth login` on the machine running the scripts. The runner refuses public destination datasets. A complete CUDA job requires server storage for dependencies, models, the source dataset and temporary batches; check your allocation and quota first.

Example, from the repository root, using a GPU allocated to you:

```sh
export CUDA_VISIBLE_DEVICES=1
uv run --locked --script src/experiments/cloud_prefill.py \
  --repo tandemkv/paired-kv-qwen-smollm \
  --prefix runs/your-unique-run --work /path/to/private/scratch \
  --count 200 --batch 5
```

Select a unique prefix and an empty work directory. To resume after an interruption, use the same arguments and work directory. The runner verifies uploaded file hashes at each checkpoint before removing that batch. `complete.json` records completion; `last_upload.json` records the corresponding immutable revision. Do not run two processes against the same prefix.

For the existing 200-prompt dataset, the replay check covers five training and five validation prompts sampled across lengths for both models:

```sh
uv run --locked --script src/experiments/replay_batch.py --work /path/to/private/scratch --device cuda
```

To create labels for a manifest already downloaded to a work directory:

```sh
uv run --locked --script src/experiments/paired_kv_dataset.py labels \
  --output /path/to/manifest-directory \
  --dataset-repo tandemkv/paired-kv-qwen-smollm \
  --dataset-revision dca861cd61b10859d5682565cfbba5606e5c5974 \
  --manifest-path runs/hotpot-200-20261028/manifest.json
```

Upload the resulting `labels.json` alongside that manifest after verifying it. The recorded 200-prompt labels use the source revision and manifest identity in the file; they do not require downloading the feature tensors. Native QA evaluation is covered separately by PR #53. This dataset is an initial HotpotQA-only experiment, not HeteroFold's full calibration corpus or evidence of mapper quality.
