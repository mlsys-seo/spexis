# Spexis

**Speculative and Lookahead Scheduling for LLM Inference**

Spexis is a multi-GPU LLM inference framework that improves the
efficiency of pipeline and tensor parallelism through *speculative
parallelism*. Rather than using speculative decoding only to accelerate
token generation, Spexis runs speculation in parallel with normal
execution — a small early-exit (draft) model attached to an early
pipeline stage speculates the next token while the rest of the target
model is still computing, and the speculation is verified one pipeline
iteration later — introducing a new parallelism axis **without
increasing KV-cache memory usage**.

Spexis further uses *lookahead scheduling* to predict speculation
quality and future memory pressure, reducing wasted speculation,
KV-cache eviction, and recomputation. Built on top of
[vLLM](https://github.com/vllm-project/vllm) v0.8.4, Spexis achieves
speedups of up to 34% over a baseline that uses the optimal combination
of pipeline and tensor parallelism.

Spexis is a research prototype released alongside the paper, not a
production serving stack; only the parts needed for the paper's
experiments are kept (CUDA, Llama/Qwen families).

📄 [**Spexis: Speculative and Lookahead Scheduling for LLM Inference**](https://arxiv.org/abs/2609.34370) — to appear at EMNLP 2026 main.

## Requirements

- NVIDIA GPUs with CUDA 12.4. Tested on A40, H100 SXM, A100 SXM,
  and L40S.
- Two or more GPUs — speculative parallelism requires
  `pipeline_parallel_size >= 2`.
- Python 3.11, PyTorch 2.6.0. Tested inside the
  `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` container; the setup
  below is everything needed on top of that image.

## Setup

Spexis is a **Python-only fork** of vLLM v0.8.4: instead of building
CUDA kernels from source, the install opens the official `vllm==0.8.4`
wheel and extracts its compiled kernels into the `spexis/` package.

```bash
git clone https://github.com/mlsys-seo/spexis.git && cd spexis

# 1. Build tooling
pip install -U pip setuptools wheel packaging setuptools-scm jinja2 ninja cmake

# 2. Fetch the upstream wheel Spexis links against (no build happens)
pip download --no-deps "vllm==0.8.4" -d /tmp/vllm-wheel
export VLLM_PRECOMPILED_WHEEL_LOCATION=$(ls /tmp/vllm-wheel/vllm-0.8.4*.whl)
export VLLM_USE_PRECOMPILED=1

# 3. Install Spexis and its pinned runtime deps
pip install -e . --no-build-isolation
pip install "transformers==4.57.6" "ray[cgraph]==2.53.0" accelerate

# 4. Verify
python -c "import spexis; print(spexis.__version__)"
```

The package installs and imports as `spexis`, so it can coexist with an
upstream vLLM installation. Environment variables keep their `VLLM_*`
names and the torch custom-op namespace remains `torch.ops.vllm`, so
existing launch scripts and tooling keep working.

### Exit-model artifacts

Speculation needs two artifacts per target model:

- `exit_1_<pp>/` — the trained early-exit model for that pipeline split
  (`exit_1_2` for PP=2, `exit_1_4` for PP=4). Not in this repository
  yet; release is in progress.
- `embed_tokens.bin` — the target model's embedding weights, which the
  first stage looks up on the CPU instead of holding on the GPU. This is
  part of the target model's own weights, so it is not distributed;
  generate it from the checkpoint you downloaded:

  ```bash
  python tools/extract_embed_tokens.py \
      --model /path/to/Llama-3.3-70B-Instruct \
      --output /path/to/exitmodels/Llama-3.3-70B-Instruct/embed_tokens.bin
  ```

## Quick start

Offline inference with speculative parallelism on two GPUs:

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

The same options are available on the OpenAI-compatible server:

```bash
spexis serve /path/to/Llama-3.3-70B-Instruct \
    --pipeline-parallel-size 2 --distributed-executor-backend ray \
    --enable-spexis True \
    --exit-layer True False \
    --exit-models /path/to/exitmodels/.../exit_1_2 None \
    --cpu-embedding-path /path/to/exitmodels/.../embed_tokens.bin \
    --exit-model-type llama --verify-strategy greedy
```

Pass one `--exit-layer` flag and one `--exit-models` entry per pipeline
stage; stages that do not speculate take the literal string `None`.

Key options:

| Option | Meaning |
|---|---|
| `--enable-spexis` | Turn on speculative parallelism |
| `--spexis-ver` | `1.0` speculative parallelism only; `2.0` + confidence-based drop; `2.5` + length-aware scheduling (needs `--length-predictor-path`). Cumulative; the paper uses `2.5` |
| `--exit-layer` / `--exit-models` | Per-stage: does this stage run an exit model, and with which weights |
| `--cpu-embedding-path` | Target-model embedding weights, looked up on the CPU |
| `--exit-model-type` | `llama` or `qwen` |
| `--drop-ratio` | Fraction of the batch dropped from speculation each iteration, lowest confidence first (`--spexis-ver` >= 2.0) |
| `--enable-log`, `--logdir` | Write the per-iteration metrics log (JSONL) |

## Reproducing the paper

The serving and fixed-batch throughput experiments are scripted under
[`experiments/`](experiments/README.md):

```bash
cd experiments/serving   && ./run_serving.sh       # end-to-end serving comparison
cd experiments/throughput && ./run_fixed_batch.sh  # saturated throughput at fixed batch sizes
```

See `experiments/README.md` for datasets, knobs, and output formats.

## Citation

*Spexis: Speculative and Lookahead Scheduling for LLM Inference* is to
appear at EMNLP 2026. The paper link and BibTeX entry will be added here
once the proceedings are published.

## Acknowledgment

This repository is a fork of the
[vLLM project](https://github.com/vllm-project/vllm) at v0.8.4; support
for non-CUDA platforms, most model architectures, and V0 speculative
decoding was removed to keep the research code readable. Where upstream
files are modified, the changed regions are marked with `[Spexis]`. See
`NOTICE` for the list of modifications.

This is a refactored version of the code used in the Spexis paper; we thank
the contributors to the original implementation:
[@hyungyuJung](https://github.com/hyungyuJung), [@Jaehyuk Yu](https://github.com/my-snow), [@Hoonseo Choi](https://github.com/hoonseo0316),
and [@Sungkyun Kim](https://github.com/SungBalance).

Licensed under Apache-2.0, the same license as vLLM. See `LICENSE`.
