
# GRPO

GPU benchmark training a small causal LM with **Group Relative Policy
Optimization** via HuggingFace TRL's `GRPOTrainer`. Mirrors the
neighbouring `rlhf` (PPO) bench in dataset, model, and instrumentation
so the two are directly comparable.

| Setting | Value |
| --- | --- |
| Trainer | `trl.GRPOTrainer` |
| Policy / reward model | `EleutherAI/pythia-1b-deduped` (one checkpoint, two roles) |
| Dataset | `trl-internal-testing/descriptiveness-sentiment-trl-style` (`descriptiveness` split, `prompt` column only) |
| Generation | in-process HF `.generate()` for the base variants; co-located vLLM for the `*-vllm-*` variants |

Group-relative advantages replace PPO's learned critic, so no value
model is loaded.

## Variants

- `grpo-single` / `grpo-gpus` — pure HF generation.
- `grpo-vllm-single` / `grpo-vllm-gpus` — same setup, generation
  off-loaded to a co-located vLLM engine
  (`vllm_mode: colocate`, `vllm_gpu_memory_utilization: 0.15`,
  `vllm_max_model_length: 1024`).

## Local dev

```bash
cd benchmarks/grpo
milabench install --config dev.yaml --base .
milabench prepare --config dev.yaml --base .
milabench run     --config dev.yaml --base .
```

Use `--select grpo-single`, `--select grpo-vllm-single`, etc. to run a
single variant.
