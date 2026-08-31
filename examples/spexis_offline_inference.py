# SPDX-License-Identifier: Apache-2.0
"""Minimal Spexis example: speculative pipeline parallelism, offline.

Every stage but the last runs an early-exit (draft) model in parallel with
the target model, so speculation costs no extra KV cache. Pass one
--exit-layer flag and one --exit-models entry per pipeline stage; stages
without an exit model take the literal string "None".

    python examples/spexis_offline_inference.py \
        --model /path/to/Llama-3.3-70B-Instruct \
        --exit-model-dir /path/to/exitmodels/Llama-3.3-70B-Instruct

The exit-model directory must contain the trained exit model for the
split (exit_1_2 for PP=2, exit_1_4 for PP=4) and embed_tokens.bin, the
target model's embedding weights (the first stage looks embeddings up on
the CPU instead of holding them on the GPU).
"""

import argparse
import os

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Target model path.")
    parser.add_argument("--exit-model-dir", required=True,
                        help="Directory holding exit_1_<pp> and "
                             "embed_tokens.bin for this target model.")
    parser.add_argument("--pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--exit-model-type", default="llama",
                        choices=["llama", "qwen"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--baseline", action="store_true",
                        help="Run plain pipeline parallelism instead, for "
                             "comparison.")
    args = parser.parse_args()

    pp_size = args.pipeline_parallel_size
    engine_kwargs = dict(
        model=args.model,
        pipeline_parallel_size=pp_size,
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_executor_backend="ray",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    if not args.baseline:
        exit_model = os.path.join(args.exit_model_dir, f"exit_1_{pp_size}")
        embeddings = os.path.join(args.exit_model_dir, "embed_tokens.bin")
        for path in (exit_model, embeddings):
            if not os.path.exists(path):
                raise SystemExit(f"missing exit-model artifact: {path}")
        engine_kwargs.update(
            enable_specpipe=True,
            # Stage 0 speculates; later stages only verify.
            exit_layer=[True] + [False] * (pp_size - 1),
            exit_models=[exit_model] + ["None"] * (pp_size - 1),
            cpu_embedding_path=embeddings,
            exit_model_type=args.exit_model_type,
            verify_strategy="greedy",
            drop_ratio=0.0,
        )

    llm = LLM(**engine_kwargs)
    prompts = [
        "The capital of France is",
        "Speculative decoding accelerates inference by",
    ]
    outputs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    for out in outputs:
        print("-" * 70)
        print("prompt:", out.prompt)
        print("output:", out.outputs[0].text)


if __name__ == "__main__":
    main()
