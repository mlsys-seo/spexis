# SPDX-License-Identifier: Apache-2.0
"""POSIX shared-memory channel between the engine core and the stage-0 worker.

The exit-layer (spec) output cannot travel back through the Ray compiled DAG
(the DAG output slot is taken by the true-output path), so the stage-0 worker
publishes it out-of-band through a single shared-memory segment:

    [ flag : 4B int32 ][ len : 4B int32 ][ payload : pickled bytes ]

Writer (``ray_utils.execute_specpipe_model_ray``): spin-waits until
``flag == 0``, writes header + payload, then sets ``flag = 1``.
Reader (engine core via :func:`read_result_from_shm`): spin-waits until
``flag == 1``, copies the payload out, then resets ``flag = 0``.

Spin-waiting is deliberate: the hand-off sits on the critical path of every
pipeline iteration, so sub-microsecond wake-up is preferred over a
sleep/condvar at the cost of keeping one CPU core busy during the wait.
"""

import numpy as np

SHM_NAME = "exitlayer_shared_mem"
# Maximum pickled spec-output size. The writer asserts the payload fits
# (see ray_utils.execute_specpipe_model_ray); grow this if batches get
# large enough to overflow 1 MiB.
MAX_SIZE = 1024 * 1024
FLAG_SIZE = 4
HEADER_SIZE = 4


def read_result_from_shm(np_shm_buffer: np.ndarray) -> bytes:
    """Blocking read of one message from the shared-memory channel."""
    while True:
        flag = np.frombuffer(np_shm_buffer[0:FLAG_SIZE], dtype=np.int32)[0]
        if flag == 1:
            data_len = np.frombuffer(
                np_shm_buffer[FLAG_SIZE:FLAG_SIZE + HEADER_SIZE],
                dtype=np.int32)[0]
            payload = np_shm_buffer[FLAG_SIZE + HEADER_SIZE:FLAG_SIZE +
                                    HEADER_SIZE + data_len].tobytes()
            # Release the slot for the writer.
            np_shm_buffer[0:FLAG_SIZE] = np.array(
                [0], dtype=np.int32).view(np.uint8)
            return payload
