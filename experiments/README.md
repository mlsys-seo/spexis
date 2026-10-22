# Experiments

Scripts to reproduce the paper's main results. Both scripts are
self-contained: every knob is an environment variable with a sensible
default, listed at the top of each script.

## Prerequisites

Spexis installed (see the top-level README) on a multi-GPU node, plus
the benchmark harness's extra dependencies:

```bash
pip install pandas
```

You also need the models and a dataset below. All paths are passed to
the scripts as environment variables, so the files can live anywhere as
long as the directory *structure* matches.

### Models and artifacts

```
$MODEL_DIR                 # target model: a standard HF checkpoint
                           #   (e.g. meta-llama/Llama-3.3-70B-Instruct, Qwen/Qwen3-32B)

$EXIT_MODEL_DIR            # e.g. /path/to/spexis-weight/exit-models/Llama-3.3-70B-Instruct/
  exit_1_2/                # early-exit (draft) model for the PP=2 split
    config.json
    pytorch_model.bin
  exit_1_4/                # same structure, trained for the PP=4 split

$LENGTH_PREDICTOR_DIR      # SPEXIS_VER=2.5 only, e.g.
                           #   /path/to/spexis-weight/length-predictors/Llama-3.3-70B-Instruct/
```

**Exit models.** Each `exit_1_<pp>/` directory is a self-contained
model in standard HF format: `config.json` is a `*ForCausalLM` config
with `num_hidden_layers = 1` and the target model's hidden dimensions,
and `pytorch_model.bin` holds one decoder layer plus an lm_head
(~2.8 GB for the 70B target). It consumes the hidden states at the
first pipeline-stage boundary, so it is trained per target model *and*
per split point (`exit_1_2` = after 1/2 of the layers, `exit_1_4` =
after 1/4); the directories are not interchangeable. The scripts pick
`exit_1_2` or `exit_1_4` automatically from the PP size.

The first pipeline stage looks token embeddings up on the CPU instead
of holding a GPU copy; the engine reads that table straight from
`$MODEL_DIR` (only the shard holding `model.embed_tokens.weight`).

**Length predictor** (`SPEXIS_VER=2.5`). A small MLP over the exit
layer's hidden state that predicts the probability that a request's
remaining length exceeds each threshold. Like the exit models it is
trained per target and per split, laid out as:

```
length-predictors/
  Llama-3.3-70B-Instruct/
    exit_1_2/
      config.json          # in_dim, hidden, thresholds_list, dropout_p
      model.safetensors
    exit_1_4/
  Qwen3-32B/
```

Like `EXIT_MODEL_DIR`, `LENGTH_PREDICTOR_DIR` points at the
model-level directory and the scripts append `exit_1_2`/`exit_1_4`
from the PP size. (The engine flag `--length-predictor-path` itself
takes the split-level directory, if you launch the engine directly.)

### Datasets

`DATASET` selects the loader and `DATASET_PATH` the file:

| `DATASET` | Expected file | Source |
|---|---|---|
| `specbench` | JSONL, one `{question_id, category, turns, reference}` per line | [Spec-Bench](https://github.com/hemingkx/Spec-Bench) `data/spec_bench/question.jsonl` |
| `sharegpt` | single JSON array of `{conversations: [{from, value}, ...]}` | `ShareGPT_V3_unfiltered_cleaned_split.json` (HF) |
| `lmsyschat` | parquet | `lmsys/lmsys-chat-1m` |

The paper mainly uses SpecBench for throughput and SpecBench /
LMSYS-Chat for serving.

## 1. Serving experiment (`serving/run_serving.sh`)

The main end-to-end comparison: an OpenAI-compatible server is launched
for each configuration, and a client submits requests as a Poisson
process at a set of request rates for a fixed duration.

```bash
cd experiments/serving
MODEL_DIR=/path/to/Llama-3.3-70B-Instruct \
EXIT_MODEL_DIR=/path/to/spexis-weight/exit-models/Llama-3.3-70B-Instruct \
DATASET_PATH=/path/to/spec_bench/question.jsonl \
PP_TP_PAIRS="2,4" SPEXIS_OPTS="True False" SPEXIS_VER=1.0 \
REQUEST_RATES="3.8 4.0 4.2" \
./run_serving.sh
```

Each run writes `client.log` (latency/throughput summary),
`server_console.log`, and the per-iteration server log
(`server_iter_log.jsonl`) under
`<repo-parent>/cache/spexis_results/serving/<experiment-id>/`
(override with `RESULT_ROOT`).
The baseline runs (`SPEXIS_OPTS` containing `False`) use plain PP/TP
with the same engine settings.

## 2. Fixed-batch throughput experiment (`throughput/run_fixed_batch.sh`)

Measures saturated decode throughput at a fixed batch size N: all N
requests enter the engine at once with EOS ignored, so the running
batch stays at N, and `log_iter/throughput_saturated*.py` measures
tokens/s over a window of 50 consecutive full-batch iterations from the
per-iteration log (for Spexis runs it also reports the speculation
accuracy).

```bash
cd experiments/throughput
MODEL_DIR=/path/to/Llama-3.3-70B-Instruct \
EXIT_MODEL_DIR=/path/to/spexis-weight/exit-models/Llama-3.3-70B-Instruct \
DATASET_PATH=/path/to/spec_bench/question.jsonl \
PP_TP_PAIRS="2,1" BATCH_SIZES="16 32 64" \
./run_fixed_batch.sh
```

Notes:

- With `DROP_RATIO > 0` the analyzer measures the window at the
  effective batch size `N - floor(N * drop_ratio)`.

## Layout

```
experiments/
  benchmark/    client and offline harnesses (serving duration, throughput)
  log_iter/     per-iteration-log analyzers (saturated throughput)
  serving/      experiment 1
  throughput/   experiment 2
```
