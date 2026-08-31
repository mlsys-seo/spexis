# Spexis

**Speculative and Lookahead Scheduling for LLM Inference**

Spexis is a multi-GPU LLM inference framework that turns speculation into a
parallelism axis. Instead of using speculative decoding only to accelerate
token generation, Spexis runs speculation *in parallel with* normal
execution: every pipeline stage but the last carries a small early-exit
(draft) model that speculates while the target model computes, and the
speculation is verified one pipeline iteration later. Because the
speculative tokens ride along with the existing batch, this adds
parallelism **without increasing KV-cache memory usage**.

On top of that, Spexis uses *lookahead scheduling*: it predicts speculation
quality and future memory pressure to decide which requests should keep
speculating and when new prefills can be admitted, which reduces wasted
speculation, KV-cache eviction, and recomputation.

Spexis is built on [vLLM](https://github.com/vllm-project/vllm) v0.8.4. The
paper reports 16–34% speedup over a baseline using the best combination of
pipeline and tensor parallelism.

📄 **Spexis: Speculative and Lookahead Scheduling for LLM Inference** — to appear at EMNLP 2026.

---

## How it works

| Component | Where |
|---|---|
| Speculative pipeline scheduler (drop policy, lookahead admission) | `vllm/specpipe/specpipe_scheduler.py` |
| Early-exit draft models (Llama / Qwen adapters) | `vllm/specpipe/exit_models/` |
| Model runner that runs target model + exit layer together | `vllm/specpipe/gpu_model_runner.py` |
| Generation-length predictor and peak-KV projection | `vllm/specpipe/length_predictor.py`, `length_calc.py` |
| KV manager that keeps unverified tokens out of the prefix cache | `vllm/specpipe/specpipe_kv_cache_manager.py` |
| Per-iteration metrics log | `vllm/specpipe/iteration_log.py` |

Everything Spexis adds lives in `vllm/specpipe/`. Where upstream vLLM files
had to be touched, the changed regions are marked with `[Spexis]`, so
`grep -rn "\[Spexis\]" vllm/` shows the full integration surface.

## Requirements

- NVIDIA GPU, CUDA 12.4 (tested on H100 NVL, A100, L40S)
- Python 3.11, PyTorch 2.6.0
- Two or more GPUs (speculative pipelining requires `pipeline_parallel_size >= 2`)

Tested inside the `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` container.
Spexis ships no Dockerfile: the install below is all that is needed on top
of that image.

## Install

Spexis is a **Python-only fork**: it reuses the compiled kernels from the
official vLLM v0.8.4 wheel instead of building its own.

```bash
git clone <this-repo> && cd spexis
pip install -U pip setuptools wheel packaging setuptools-scm jinja2 ninja cmake

# Fetch the upstream wheel Spexis links against
pip download --no-deps "vllm==0.8.4" -d /tmp/vllm-wheel
export VLLM_PRECOMPILED_WHEEL_LOCATION=$(ls /tmp/vllm-wheel/vllm-0.8.4*.whl)
export VLLM_USE_PRECOMPILED=1

pip install -e . --no-build-isolation
pip install "transformers==4.57.6" "ray[cgraph]==2.53.0" accelerate
```

> The package is still importable as `vllm`, so a Spexis install replaces
> vLLM in the same environment. Use a dedicated virtualenv/conda env.

### Exit-model artifacts

Speculation needs two files per target model, which are **not** in this
repository yet:

- `exit_1_<pp>/` — the trained early-exit model for that pipeline split
- `embed_tokens.bin` — the target model's embedding weights, which the
  first stage looks up on the CPU instead of holding on the GPU

*(release of these artifacts is in progress; see Status below)*

## Quick start

```bash
python examples/spexis_offline_inference.py \
    --model /path/to/Llama-3.3-70B-Instruct \
    --exit-model-dir /path/to/exitmodels/Llama-3.3-70B-Instruct \
    --pipeline-parallel-size 2

# Plain pipeline parallelism, for comparison
python examples/spexis_offline_inference.py \
    --model /path/to/Llama-3.3-70B-Instruct \
    --exit-model-dir /path/to/exitmodels/Llama-3.3-70B-Instruct \
    --baseline
```

The same options are available on the engine and the OpenAI-compatible
server, e.g.:

```bash
vllm serve /path/to/Llama-3.3-70B-Instruct \
    --pipeline-parallel-size 2 --distributed-executor-backend ray \
    --enable-specpipe True \
    --exit-layer True False \
    --exit-models /path/to/exitmodels/.../exit_1_2 None \
    --cpu-embedding-path /path/to/exitmodels/.../embed_tokens.bin \
    --exit-model-type llama --verify-strategy greedy --drop-ratio 0.0
```

Pass one `--exit-layer` flag and one `--exit-models` entry per pipeline
stage; stages that do not speculate take the literal string `None`.

Key options:

| Option | Meaning |
|---|---|
| `--enable-specpipe` | Turn on speculative pipelining |
| `--exit-layer` | Per-stage flag: does this stage run an exit model? |
| `--exit-models` | Per-stage exit model path (`None` if the stage has no exit model) |
| `--cpu-embedding-path` | Target model embedding weights, looked up on the CPU |
| `--exit-model-type` | `llama` or `qwen` |
| `--drop-ratio` | Fraction of the batch dropped from speculation each iteration, lowest confidence first |
| `--sppp-ver` | `2.0`/`2.5` enable lookahead scheduling (needs `--length-predictor-path`) |
| `--enable-log`, `--logdir` | Write the per-iteration metrics log (JSONL) |

### Troubleshooting

- **NCCL fails to start in a container**: containers often default to a
  64 MB `/dev/shm`. Mount a larger one (`--shm-size=16g`, or a
  `Memory`-medium `emptyDir` on Kubernetes). `NCCL_SHM_DISABLE=1` works
  around it at a performance cost.
- **H100 + NCCL "unhandled cuda error"**: set `NCCL_NVLS_ENABLE=0`.

## Status

This is a research prototype released alongside the paper, not a
production serving stack. It is being cleaned up in stages:

- [x] Core system (`vllm/specpipe/` and its engine integration)
- [x] Trimmed to the CUDA-only surface the paper uses (Llama/Qwen families)
- [ ] Experiment and reproduction scripts
- [ ] Exit-model artifacts and the code that trains them
- [ ] Package rename (`vllm` → `spexis`)

Feature parity with upstream vLLM is explicitly *not* a goal: support for
non-CUDA platforms, most model architectures, and V0 speculative decoding
has been removed to keep the research code readable.

## Citation

*Spexis: Speculative and Lookahead Scheduling for LLM Inference* is to
appear at EMNLP 2026. The paper link and BibTeX entry will be added here
once the proceedings are published.

## Acknowledgment

This repository is a fork of the [vLLM project](https://github.com/vllm-project/vllm)
at v0.8.4. Spexis is a research prototype and does not have feature parity
with upstream vLLM; only the parts needed for the paper's experiments were
retained. See `NOTICE` for the list of modifications.

Licensed under Apache-2.0, the same license as vLLM. See `LICENSE`.
