# SPDX-License-Identifier: Apache-2.0
"""Per-iteration JSONL metrics log for the engine core.

Throughput analysis (see ``exp_utils/log_iter``) needs one record per
engine-core iteration. :class:`IterationLogger` owns the output file and
hands out one :class:`IterationRecord` per iteration through a context
manager, so a record is always written exactly once -- including on the
early-return paths -- and the file is opened once rather than reopened on
every iteration.

Record schema (one JSON object per line)::

    {
      "iter": 12,                       # monotonic iteration index
      "time": 1234.5678,                # perf_counter at iteration start
      "sched": {"tokens": 512,          # tokens scheduled this iteration
                "new_reqs": 2,          # newly scheduled (prefill) requests
                "cached_reqs": 30},     # continued (decode) requests
      "spec": {"correct": 20,           # verified speculative tokens
               "wrong": 3,              # rejected -> rolled back
               "flush_norm": 1,         # flushed after a correction
               "flush_drop": 0,         # flushed while dropped
               "final_drop": 0},        # dropped at the end of a run
      "cpu_embed_s": {"update_states": 1.1e-4,   # CPU embedding stage times
                      "reorder": 2.2e-4,
                      "lookup": 3.3e-4},
      "len_estimate_s": 0.0012,         # lookahead memory estimation cost
      "max_dropped_confidence": 0.45    # highest confidence that was dropped
    }

``sched`` is always present (zeros when nothing was scheduled); the
speculation-only fields are omitted for non-specpipe runs.
"""

import json
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence

from vllm.logger import init_logger

logger = init_logger(__name__)


class IterationRecord:
    """Metrics collected during a single engine-core iteration."""

    __slots__ = ("_data", )

    def __init__(self, index: int):
        self._data: dict[str, Any] = {
            "iter": index,
            "time": time.perf_counter(),
            "sched": {
                "tokens": 0,
                "new_reqs": 0,
                "cached_reqs": 0
            },
        }

    def set_schedule(self, scheduler_output) -> None:
        """Record what the scheduler admitted this iteration."""
        self._data["sched"] = {
            "tokens": scheduler_output.total_num_scheduled_tokens,
            "new_reqs": len(scheduler_output.scheduled_new_reqs),
            "cached_reqs": len(scheduler_output.scheduled_cached_reqs),
        }

    def set_spec_results(self, counts: Sequence[int]) -> None:
        """Record verification outcomes (see specpipe_scheduler result codes)."""
        correct, wrong, flush_norm, flush_drop, final_drop = counts
        self._data["spec"] = {
            "correct": correct,
            "wrong": wrong,
            "flush_norm": flush_norm,
            "flush_drop": flush_drop,
            "final_drop": final_drop,
        }

    def set_cpu_embed_times(self, update_states: float, reorder: float,
                            lookup: float) -> None:
        """Record the CPU-side embedding pipeline stage timings (seconds)."""
        self._data["cpu_embed_s"] = {
            "update_states": update_states,
            "reorder": reorder,
            "lookup": lookup,
        }

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def to_json(self) -> str:
        return json.dumps(self._data)


class IterationLogger:
    """Writes one :class:`IterationRecord` per iteration to a JSONL file."""

    def __init__(self, path: Optional[str], enabled: bool = False):
        self.enabled = bool(enabled and path)
        self.path = path
        self.iter = 0
        self._file = None
        if enabled and not path:
            logger.warning("Iteration logging requested without --logdir; "
                           "no iteration log will be written.")

    @contextmanager
    def record(self) -> Iterator[IterationRecord]:
        """Yield a record for this iteration and write it on exit."""
        rec = IterationRecord(self.iter)
        try:
            yield rec
        finally:
            self.iter += 1
            if self.enabled:
                self._write(rec)

    def _write(self, rec: IterationRecord) -> None:
        if self._file is None:
            self._file = open(self.path, "a", encoding="utf-8")
        self._file.write(rec.to_json())
        self._file.write("\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
