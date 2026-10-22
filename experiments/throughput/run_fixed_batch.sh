#!/bin/bash
# Fixed-batch-size throughput experiment (paper Sec. 7.2).
#
# For each batch size N, the offline engine is loaded with N requests at
# once with EOS ignored, so the batch stays full; the per-iteration log
# is then analyzed over a window of 50 consecutive full-batch iterations
# to report saturated decode throughput (tokens/s) and, for Spexis runs,
# the speculation accuracy.
#
# Every knob below can be overridden from the environment, e.g.
#   BATCH_SIZES="16 32 64" SPEXIS_OPTS="True" ./run_fixed_batch.sh
set -u

# ----------------------------- configuration -----------------------------
MODEL_DIR=${MODEL_DIR:-/workspace/models/Llama-3.3-70B-Instruct}
EXIT_MODEL_DIR=${EXIT_MODEL_DIR:-/workspace/models/exitmodels/Llama-3.3-70B-Instruct}
EXIT_MODEL_TYPE=${EXIT_MODEL_TYPE:-llama}          # llama | qwen
LENGTH_PREDICTOR_DIR=${LENGTH_PREDICTOR_DIR:-}     # required for SPEXIS_VER 2.5;
                                                   # model-level dir, split picked by PP

DATASET=${DATASET:-specbench}
DATASET_PATH=${DATASET_PATH:-/workspace/cache/specbench_question.jsonl}

# Results go next to (not inside) the repository:
#   <repo-parent>/cache/spexis_results/throughput
REPO_PARENT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
RESULT_ROOT=${RESULT_ROOT:-$REPO_PARENT/cache/spexis_results/throughput}

PP_TP_PAIRS=${PP_TP_PAIRS:-"2,1"}                  # space-separated "pp,tp" pairs
SPEXIS_OPTS=${SPEXIS_OPTS:-"True False"}           # run Spexis and/or the PP/TP baseline
SPEXIS_VER=${SPEXIS_VER:-1.0}                      # 1.0 | 2.0 | 2.5 (see README)
DROP_RATIO=${DROP_RATIO:-0.0}
BATCH_SIZES=${BATCH_SIZES:-"16 32 64"}             # fixed batch sizes to sweep

TEMPERATURE=${TEMPERATURE:-0.0}
MAX_SEQS=${MAX_SEQS:-512}
TOKEN_BUDGET=${TOKEN_BUDGET:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}              # True disables CUDA graph capture
# -------------------------------------------------------------------------

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS="$SCRIPT_DIR/../benchmark/benchmark_throughput.py"
ANALYZER_DIR="$SCRIPT_DIR/../log_iter"
MODEL_NAME=$(basename "$MODEL_DIR")

export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=nccl
mkdir -p "$RESULT_ROOT"
TMP_LOG_PATH="$RESULT_ROOT/.tmp_iter_log.jsonl"

# Spexis implements greedy verification only (temperature 0.0).
if [[ "$SPEXIS_OPTS" == *True* && "$TEMPERATURE" != "0.0" ]]; then
    echo "Error: Spexis runs require TEMPERATURE=0.0 (greedy verification only)." >&2
    exit 1
fi
verify_strategy="greedy"

for pair in $PP_TP_PAIRS; do
    IFS=',' read -r pp_size tp_size <<< "$pair"

    case "$pp_size" in
        2) exit_layer_flags="True False"
           exit_models_args="$EXIT_MODEL_DIR/exit_1_2 None"
           lenpred_path="$LENGTH_PREDICTOR_DIR/exit_1_2" ;;
        4) exit_layer_flags="True False False False"
           exit_models_args="$EXIT_MODEL_DIR/exit_1_4 None None None"
           lenpred_path="$LENGTH_PREDICTOR_DIR/exit_1_4" ;;
        *) exit_layer_flags=""; exit_models_args=""; lenpred_path="" ;;
    esac

    for enable_spexis in $SPEXIS_OPTS; do
        if [[ "$enable_spexis" == "True" && "$pp_size" -lt 2 ]]; then
            echo "Skipping: Spexis requires pipeline_parallel_size >= 2."
            continue
        fi

        sp_tag="SpexisFalse"
        [[ "$enable_spexis" == "True" ]] && sp_tag="SpexisTrue-v${SPEXIS_VER}"

        for batch_size in $BATCH_SIZES; do
            EXP_ID="${MODEL_NAME}_PP${pp_size}_TP${tp_size}_${sp_tag}_N${batch_size}_Drop${DROP_RATIO}"
            EXP_DIR="$RESULT_ROOT/$EXP_ID"
            mkdir -p "$EXP_DIR"
            if [[ -f "$EXP_DIR/result_log.jsonl" ]]; then
                echo "Skipping $EXP_ID (already exists)"; continue
            fi
            echo "=================================================="
            echo "Experiment: $EXP_ID"
            echo "=================================================="
            rm -f "$TMP_LOG_PATH"

            CMD=( python "$HARNESS"
                --backend vllm
                --model "$MODEL_DIR"
                --dataset-name "$DATASET" --dataset-path "$DATASET_PATH"
                --num-prompts "$batch_size"
                --temperature "$TEMPERATURE"
                --max-num-seqs "$MAX_SEQS"
                --max-num-batched-tokens "$TOKEN_BUDGET"
                --max-model-len "$MAX_MODEL_LEN"
                --pipeline_parallel_size "$pp_size"
                --tensor_parallel_size "$tp_size"
                --distributed-executor-backend ray
                --enable-log True --logdir "$TMP_LOG_PATH"
            )
            [[ "$ENFORCE_EAGER" == "True" ]] && CMD+=( --enforce-eager )

            if [[ "$enable_spexis" == "True" ]]; then
                CMD+=( --enable-spexis True
                    --spexis-ver "$SPEXIS_VER"
                    --verify-strategy "$verify_strategy"
                    --drop-ratio "$DROP_RATIO"
                    --exit-model-type "$EXIT_MODEL_TYPE"
                    --exit-layer $exit_layer_flags
                    --exit-models $exit_models_args
                )
                if [[ "$SPEXIS_VER" == "2.5" ]]; then
                    CMD+=( --length-predictor-path "$lenpred_path" )
                fi
            fi

            "${CMD[@]}" > "$EXP_DIR/console.log" 2>&1 \
                || { echo "!! Run failed; see $EXP_DIR/console.log"; continue; }

            # Saturated-window analysis over the iteration log.
            if [[ "$enable_spexis" == "True" ]]; then
                python "$ANALYZER_DIR/throughput_saturated_sp.py" \
                    --temp-log-path "$TMP_LOG_PATH" \
                    --batch-size "$batch_size" --drop-ratio "$DROP_RATIO" \
                    | tee -a "$EXP_DIR/console.log"
            else
                analyzer="throughput_saturated_pp.py"
                [[ "$pp_size" -eq 1 ]] && analyzer="throughput_saturated_tp.py"
                python "$ANALYZER_DIR/$analyzer" \
                    --temp-log-path "$TMP_LOG_PATH" \
                    --batch-size "$batch_size" --pp-size "$pp_size" \
                    | tee -a "$EXP_DIR/console.log"
            fi

            mv -f "$TMP_LOG_PATH" "$EXP_DIR/result_log.jsonl"
            echo "Completed: $EXP_ID"
        done
    done
done

echo "All throughput experiments completed. Results in $RESULT_ROOT"
