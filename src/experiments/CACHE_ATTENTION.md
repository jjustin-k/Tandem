# Conditional cache attention: first experiment

Goal: translate SmolLM2-360M-Instruct caches into Qwen2.5-0.5B-Instruct caches and preserve Qwen's native behavior on the same saved prompt. No answer labels or prompt reformatting are used. Both LLMs remain frozen. The first checkpoint is trained in one direction; conditioning on model IDs does not imply support for unseen models or pairs.

## Design decisions

- **One causal boundary at a time.** Align using the existing manifest's character-boundary correspondence. Attention runs over source layers/heads, not across text positions. A future source state cannot help predict an earlier receiver state. Positions before the first causal match are excluded from fitting and use a train-only target-mean prior during replay.
- **Separate K/V observations, shared weights.** For each role and text position, source items are the 32 × 5 SmolLM layer/head vectors. Target queries identify the 24 × 2 Qwen layer/heads. Model, layer, normalized depth, head and role embeddings identify each item/query. K and V never attend to each other's observations.
- **Small attention residual.** One cross-attention block, width 64, four heads, FFN width 128, dropout zero. The 48,768 trainable parameters predict a correction to an affine baseline; the output projection starts at zero, so initial predictions exactly equal that baseline. These are architecture choices for this experiment, not settings validated by a prior paper.
- **Affine control.** Fit separate ridge maps for K/V at each target layer, using the proportionally matched source layer and all its KV heads. The intercept is unpenalized. This control has 1,972,224 frozen coefficients and is not a reproduction of HeteroFold's Recolor or multi-layer ridge methods.
- **Train-only scaling.** Estimate mean/std per feature using training prompts only, in float64. Clamp standard deviation below 0.001. Optimize standardized K/V MSE so high-magnitude heads/layers do not dominate by scale alone.
- **Receiver behavior matters.** After MSE training, calibrate the attention residual with native-teacher-to-translated-student logit KL plus 0.01 standardized K/V anchoring. Both receiver weights and native targets remain frozen. Four final tokens of each prompt are teacher-forced as a suffix; only the preceding cache is translated. This is a bounded logit-distillation experiment, not HeteroFold's attention/output objective.
- **Inference conventions.** Predict pre-RoPE keys and projected values, then apply receiver RoPE once at receiver positions. Target queries in the translator come from metadata and affine predictions, not receiver-prefilled context. The source LLM need not be resident during training on saved features.

The implementation currently requires the two supported architectures and equal head dimensions (64 for this pair). Different head counts and depths are supported. Arbitrary architecture support would require more extractors, positional handling, input/output adapters and multi-pair training.

## Evidence informing the design

[HeteroFold, §§3 and B.2](https://arxiv.org/html/2609.32259v2) motivates causal token alignment, content-space keys and receiver-aware supervision: low K/V reconstruction error alone is insufficient. Its corrections fold into affine maps; our attention network does not, so its runtime must be measured.

[Mixture-of-Translators, Appendix C.3](https://arxiv.org/html/2607.28979v2#A3.SS3) provides precedent for cross-attention cache translation with separate K/V processing. Its recurrent channel-window translators and expert routing differ from this per-boundary shared network; its results do not validate our model-ID conditioning scheme.

[Heo et al.](https://arxiv.org/abs/2608.03893) report pair-dependent linear-transfer quality and cases where nonlinear MLPs help. Their matched-KV, within-family experiments are not a guarantee for Qwen/SmolLM.

Community implementation evidence is mixed. [spotta85/kvbridge](https://github.com/spotta85/kvbridge) reports quality/runtime tradeoffs and distinguishes warm handoff from cold execution. [souvikDevloper/kvbridge](https://github.com/souvikDevloper/kvbridge) explicitly reports a real T4 run failing its quality gates despite faster mapping. These are authors' reported results, not our reproductions. They support including a baseline, replay control, behavior metrics and explicit timing boundaries.

## Cluster pilot, then full run

Use an allocated GPU via `CUDA_VISIBLE_DEVICES`. Authenticate to Hugging Face on the server. The full feature download is approximately 17.8 GB, plus CUDA dependencies, receiver weights and checkpoints; allow at least 35 GB of server scratch. Bulk tensors never need to be on the laptop.

First run four training and two validation prompts, one MSE epoch and one KL step:

```sh
uv run --locked --script src/experiments/train_cache_attention.py \
  --work /path/to/pilot --data-root /path/to/shared-feature-cache \
  --pilot --epochs 1 --kl-steps 1 --upload-prefix experiments/your-unique-pilot
```

The pilot must finish native replay controls and backward passes without errors. Inspect `outputs/final.json`, `outputs/metrics.jsonl` and peak allocated GPU memory before launching the larger run. The measured epoch time provides a first runtime estimate; four-prompt throughput must be scaled to 160 training / 40 validation prompts, and download, fitting, receiver calibration and evaluation add time. Overnight is a budget, not a convergence guarantee.

Then use the same feature cache and a separate output directory/prefix:

```sh
uv run --locked --script src/experiments/train_cache_attention.py \
  --work /path/to/full --data-root /path/to/shared-feature-cache \
  --epochs 20 --kl-steps 100 --hours 8 \
  --upload-prefix experiments/your-unique-full-run
```

The training stride is eight receiver tokens and batches contain 16 aligned positions. All source layer/head features at each selected boundary are retained. Validation prompts are document-disjoint from training. MSE training stops after four unimproved validation epochs; select the MSE checkpoint by validation loss. KL calibration selects a separate checkpoint by validation logit KL. Neither selection uses answer labels. Keep a separate test set before final claims.

`--hours` is a soft training budget checked between epochs and KL steps; preparation and final evaluation can extend total wall time. To resume, retain the work directory, feature cache and identical configuration, adding `--resume`. MSE resumes at epoch boundaries and KL resumes at saved step boundaries. With `--upload-prefix`, verified checkpoints are uploaded after each MSE epoch, every 25 KL steps and at completion. Local outputs are retained on upload failure. Initial runs refuse an already-used remote prefix. Restoring a lost work directory solely from cloud checkpoints requires manually downloading the outputs first.

## Reading results

`behavior-mse.json` compares both methods after reconstruction training. `final.json` compares the affine control with the selected attention checkpoint after calibration. Reports include teacher-forced suffix logit KL, top-1 agreement, the native replay control's maximum logit error, and peak PyTorch GPU allocation.

Timing is exploratory, single-GPU wall time with CUDA synchronization. Mapping measurements include CPU-source-feature copies, alignment, mapping, RoPE/cache construction and receiver suffix processing. Native timing includes receiver prefill and the same suffix. Source prefill, file I/O and tokenization are excluded. Single measurements are not sufficient to claim a production speedup; use matched warm-ups and repeated measurements before any latency claim. No full-text answer accuracy, free-running generation retention, multi-turn stability or universal-translation result is established by this experiment.
