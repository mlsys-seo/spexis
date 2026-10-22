#!/bin/bash
# Serving experiment (paper Sec. 7.1 / 7.3).
#
# For each (PP,TP) pair x {Spexis, baseline} x request rate, this script
# launches the OpenAI-compatible server, sends requests drawn from the
# dataset as a Poisson process for a fixed duration, and stores the
# client/server logs under $RESULT_ROOT/<experiment-id>/.
#
# Every knob below can be overridden from the environment, e.g.
#   MODEL_DIR=/models/Qwen3-32B EXIT_MODEL_TYPE=qwen ./run_serving.sh
set -u

# ----------------------------- configuration -----------------------------
MODEL_DIR=${MODEL_DIR:-/workspace/models/Llama-3.3-70B-Instruct}
EXIT_MODEL_DIR=${EXIT_MODEL_DIR:-/workspace/models/exitmodels/Llama-3.3-70B-Instruct}
EXIT_MODEL_TYPE=${EXIT_MODEL_TYPE:-llama}          # llama | qwen
LENGTH_PREDICTOR_DIR=${LENGTH_PREDICTOR_DIR:-}     # required for SPEXIS_VER 2.5;
                                                   # model-level dir, split picked by PP

DATASET=${DATASET:-specbench}                      # specbench | sharegpt | lmsyschat
DATASET_PATH=${DATASET_PATH:-/workspace/cache/specbench_question.jsonl}

# Results go next to (not inside) the repository:
#   <repo-parent>/cache/spexis_results/serving
REPO_PARENT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
RESULT_ROOT=${RESULT_ROOT:-$REPO_PARENT/cache/spexis_results/serving}

PP_TP_PAIRS=${PP_TP_PAIRS:-"2,1"}                  # space-separated "pp,tp" pairs
SPEXIS_OPTS=${SPEXIS_OPTS:-"True False"}           # run Spexis and/or the PP/TP baseline
SPEXIS_VER=${SPEXIS_VER:-1.0}                      # 1.0 | 2.0 | 2.5 (see README)
DROP_RATIO=${DROP_RATIO:-0.0}
REQUEST_RATES=${REQUEST_RATES:-"1.0 2.0 3.0"}      # requests/s (Poisson)

NUM_PROMPTS=${NUM_PROMPTS:-480}
BENCH_MINUTES=${BENCH_MINUTES:-15}                 # client run length per rate
TEMPERATURE=${TEMPERATURE:-0.0}
MAX_SEQS=${MAX_SEQS:-512}
TOKEN_BUDGET=${TOKEN_BUDGET:-2048}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
SEED=${SEED:-42}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}              # True disables CUDA graph capture

PORT=${PORT:-8005}
HOST="127.0.0.1"
SERVER_TIMEOUT=${SERVER_TIMEOUT:-600}              # seconds to wait for /health
# -------------------------------------------------------------------------

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CLIENT="$SCRIPT_DIR/../benchmark/benchmark_serving_duration.py"
MODEL_NAME=$(basename "$MODEL_DIR")

export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=nccl
mkdir -p "$RESULT_ROOT"

SERVER_PID=""

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]] && ps -p "$SERVER_PID" > /dev/null 2>&1; then
        echo ">> Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" || true
        sleep 5
        ps -p "$SERVER_PID" > /dev/null 2>&1 && kill -9 "$SERVER_PID" || true
    fi
    SERVER_PID=""
}
trap cleanup_server EXIT SIGINT SIGTERM

wait_server_ready() {
    local url="http://${HOST}:${PORT}/health"
    local retries=$(( SERVER_TIMEOUT / 5 ))
    echo ">> Waiting for the server at $url (timeout ${SERVER_TIMEOUT}s)..."
    for (( i=1; i<=retries; i++ )); do
        curl -s -f "$url" > /dev/null && { echo ">> Server is ready."; return 0; }
        if [[ -n "$SERVER_PID" ]] && ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
            echo "!! Server process died; check server_console.log"; return 1
        fi
        sleep 5
    done
    echo "!! Server not ready after ${SERVER_TIMEOUT}s."; return 1
}

# Spexis implements greedy verification only (temperature 0.0).
if [[ "$SPEXIS_OPTS" == *True* && "$TEMPERATURE" != "0.0" ]]; then
    echo "Error: Spexis runs require TEMPERATURE=0.0 (greedy verification only)." >&2
    exit 1
fi
verify_strategy="greedy"

for pair in $PP_TP_PAIRS; do
    IFS=',' read -r pp_size tp_size <<< "$pair"

    # Per-stage exit-model layout: the released artifacts attach one exit
    # model to the first stage (exit_1_2 for PP=2, exit_1_4 for PP=4).
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

        for request_rate in $REQUEST_RATES; do
            EXP_ID="${MODEL_NAME}_PP${pp_size}_TP${tp_size}_${sp_tag}_Drop${DROP_RATIO}_RR${request_rate}"
            EXP_DIR="$RESULT_ROOT/$EXP_ID"
            mkdir -p "$EXP_DIR"
            if [[ -f "$EXP_DIR/client.log" ]]; then
                echo "Skipping $EXP_ID (already exists)"; continue
            fi
            echo "=================================================="
            echo "Experiment: $EXP_ID"
            echo "=================================================="

            SERVE_CMD=( spexis serve "$MODEL_DIR"
                --port "$PORT" --host "$HOST"
                --max-num-seqs "$MAX_SEQS"
                --max-num-batched-tokens "$TOKEN_BUDGET"
                --gpu-memory-utilization "$GPU_MEM_UTIL"
                --max-model-len "$MAX_MODEL_LEN"
                --pipeline-parallel-size "$pp_size"
                --tensor-parallel-size "$tp_size"
                --distributed-executor-backend ray
                --no-enable-prefix-caching
                --swap-space 16
                --disable-log-requests
                --enable-log True --logdir "$EXP_DIR/server_iter_log.jsonl"
            )
            [[ "$ENFORCE_EAGER" == "True" ]] && SERVE_CMD+=( --enforce-eager )

            if [[ "$enable_spexis" == "True" ]]; then
                SERVE_CMD+=( --enable-spexis True
                    --spexis-ver "$SPEXIS_VER"
                    --verify-strategy "$verify_strategy"
                    --drop-ratio "$DROP_RATIO"
                    --exit-model-type "$EXIT_MODEL_TYPE"
                    --exit-layer $exit_layer_flags
                    --exit-models $exit_models_args
                )
                if [[ "$SPEXIS_VER" == "2.5" ]]; then
                    SERVE_CMD+=( --length-predictor-path "$lenpred_path" )
                fi
            fi

            echo ">> Launching server..."
            "${SERVE_CMD[@]}" > "$EXP_DIR/server_console.log" 2>&1 &
            SERVER_PID=$!
            sleep 10
            if ! wait_server_ready; then cleanup_server; continue; fi

            echo ">> Running client for ${BENCH_MINUTES} min at rate ${request_rate}..."
            python "$CLIENT" \
                --backend vllm \
                --model "$MODEL_DIR" \
                --port "$PORT" \
                --dataset-name "$DATASET" \
                --dataset-path "$DATASET_PATH" \
                --num-prompts "$NUM_PROMPTS" \
                --request-rate "$request_rate" \
                --max-duration $(( BENCH_MINUTES * 60 )) \
                --temperature "$TEMPERATURE" \
                --specbench-output-len 2048 \
                --seed "$SEED" \
                > "$EXP_DIR/client.log" 2>&1 \
                && echo ">> Client finished." || echo "!! Client failed; see client.log"

            cleanup_server
            echo "Completed: $EXP_ID"
        done
    done
done

echo "All serving experiments completed. Results in $RESULT_ROOT"
