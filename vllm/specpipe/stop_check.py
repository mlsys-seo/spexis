# SPDX-License-Identifier: Apache-2.0
"""Stop-condition checks for the two-phase specpipe update cycle.

Upstream's ``check_stop`` assumes the last output token is final. Under
speculative pipelining a token is appended when the exit layer produces
it (speculatively) and only becomes final once the target model verifies
it, so stopping is checked twice with different rules:

* :func:`check_stop_from_spec` runs on the speculative append. It reports
  whether the *speculated* token would end the request but leaves
  ``request.status`` untouched, because the token may still be rejected;
  the scheduler records this as a "stop drop" instead.
* :func:`check_stop_from_validation` runs after verification. It looks
  back past the ``specpipe_len`` still-unverified tokens to the token
  being verified, and commits the finished status.
"""

from vllm.v1.request import Request, RequestStatus


def check_stop_from_spec(request: Request, max_model_len: int) -> bool:
    """Would the just-speculated token stop the request? (no status change)"""
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens):
        return True

    sampling_params = request.sampling_params
    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos
            and last_token_id == request.eos_token_id):
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        return True
    return False


def check_stop_from_validation(request: Request, max_model_len: int) -> bool:
    """Does the just-verified token stop the request? (commits the status)"""
    if ((request.num_tokens - request.specpipe_len) >= max_model_len
            or (request.num_output_tokens -
                request.specpipe_len) >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    sampling_params = request.sampling_params
    # Skip the tokens that are still speculative and unverified.
    last_token_idx = request.specpipe_len + 1
    last_token_id = request.output_token_ids[-last_token_idx]

    if (not sampling_params.ignore_eos
            and last_token_id == request.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    return False
