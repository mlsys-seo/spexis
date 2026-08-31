# SPDX-License-Identifier: Apache-2.0
"""Closed-form estimates of generation length and peak KV memory.

With ``pp_stages`` pipeline stages and per-token spec acceptance
probability ``average_spec_acc``, an accepted speculation emits one token
per pipeline iteration while a rejection costs a rollback of several
iterations. ``ExpGenLenCache``/``LenProbCache`` memoize the resulting
combinatorial sums; ``calculate_peak_mem`` combines them with per-request
remaining-length predictions to project KV usage (in tokens) over future
iterations. Used by SpecPipeScheduler (sppp >= 2.0) to delay prefill
admission when the projected peak would overflow.

Indexing convention: cache tables are addressed with ``num_iter - 1`` and
``length - 1`` (iteration/length counts start at 1).
"""
from math import comb
import random as rand

from vllm.v1.request import Request


class ExpGenLenCache:
    def __init__(self, max_iter: int, pp_stages: int, average_spec_acc: float):
        self.max_iter = max_iter
        self.cache = [-1 for _ in range(max_iter+1)]
        self.pp_stages = pp_stages
        self.average_spec_acc = average_spec_acc

    def get(self, num_iter: int) -> float:
        if num_iter > self.max_iter:
            raise ValueError(f"num_iter {num_iter} exceeds max_iter {self.max_iter}")
        key = num_iter
        if self.cache[key] == -1:
            self.cache[key] = self.calculate(num_iter)
        return self.cache[key]

    def calculate(self, num_iter: int) -> float:
        max_wrong = num_iter // self.pp_stages

        expect_gen_len = 0
        for wrong in range(0,max_wrong+1):
            true = num_iter - wrong * self.pp_stages
            length = true + wrong
            expect_gen_len += length * comb(length, true) * \
                (self.average_spec_acc ** true) * ((1 - self.average_spec_acc) ** wrong)
        return expect_gen_len

class LenProbCache:
    def __init__(self, max_iter: int, pp_stages: int, average_spec_acc: float, expect_length: list[int]):
        # TODO: optimize cache key by max_iter*max_iter size list instead of dict
        self.max_iter = max_iter
        self.prob_cache = [[-1 for _ in range(max_iter)] for _ in range(max_iter)]
        self.cum_prob_cache = [[-1 for _ in range(max_iter)] for _ in range(len(expect_length))]
        self.len_list = expect_length
        self.pp_stages = pp_stages
        self.average_spec_acc = average_spec_acc

    def cum_prob_get(self, len_idx: int, num_iter: int) -> float:
        if num_iter > self.max_iter:
            raise ValueError(f"num_iter {num_iter} exceeds max_iter {self.max_iter}")
        if self.cum_prob_cache[len_idx][num_iter-1] == -1:
            self.cum_prob_cache[len_idx][num_iter-1] = self.calc_cum_prob_len(len_idx, num_iter)
        return self.cum_prob_cache[len_idx][num_iter-1]

    def calc_cum_prob_len(self, len_idx: int, num_iter: int) -> float:
        cum_prob = 0.0
        length = self.len_list[len_idx]
        for l in range(length, num_iter + 1):
            cum_prob += self.get(l, num_iter)
        return cum_prob
    
    def get(self, length: int, num_iter: int) -> float:
        if num_iter > self.max_iter:
            raise ValueError(f"num_iter {num_iter} exceeds max_iter {self.max_iter}")
        if self.prob_cache[num_iter-1][length-1] == -1:
            self.prob_cache[num_iter-1][length-1] = self.calc_prob_len(length, num_iter)
        return self.prob_cache[num_iter-1][length-1]
    
    def calc_prob_len(self, length: int, num_iter: int) -> float:
        if (num_iter - length) % (self.pp_stages-1) != 0:
            return 0.0

        wrong = (num_iter - length) // (self.pp_stages-1)
        true = length - wrong

        prob = comb(length, wrong) * (self.average_spec_acc ** true) * ((1 - self.average_spec_acc) ** wrong)

        return prob


def calculate_iter_mem(
    exp_gen_len_cache: ExpGenLenCache,
    len_prob_cache: LenProbCache,
    num_iter: int,
    sum_req_len: int,
    current_batch: list[Request],
    length_thresholds: list[int], # threshold for each tau eg) [0.5, 0.5, 0.5, 0.5, 0.5]
    expect_precision: list[float], # epsilon
    ) -> int:
    # TODO: APPLY should_preempt logic

    expect_gen_len = exp_gen_len_cache.get(num_iter)

    num_finished_reqs = 0
    sum_finished_reqlen = 0

    # Calculate Effective Batch
    for req in current_batch:
        remain_len_idx = req.expect_remain_len_idx(length_thresholds)
        
        # If remain_length is -1, the request will not be finished in max_iter (estimation)
        if remain_len_idx == -1:
            continue

        prob_len = len_prob_cache.cum_prob_get(remain_len_idx, num_iter)

        eff_prob = prob_len * expect_precision[remain_len_idx]

        # Sampling makes admission stochastic across runs; the scheduler
        # only uses the aggregate estimate, so this is a deliberate
        # Monte-Carlo approximation of the finish probability.
        sample = rand.random()

        # Sample to decide whether finished or not
        if sample < eff_prob:
            num_finished_reqs += 1
            sum_finished_reqlen += req.num_tokens

    memory = sum_req_len + (len(current_batch)-num_finished_reqs) * expect_gen_len - sum_finished_reqlen

    return memory # Computed in Tokens

def calculate_peak_mem(
    exp_gen_len_cache: ExpGenLenCache,
    len_prob_cache: LenProbCache,
    iter_list: list[int],
    current_batch: list[Request],
    length_thresholds: list[int], # threshold for each tau eg) [0.5, 0.5, 0.5, 0.5, 0.5]
    expect_precision: list[float], # epsilon
    ) -> tuple[list[int], int, int]:
    """Calculate the peak memory across iterations."""
    ## Calculate Static Values
    # Calculate Sigma(current_length)
    sum_req_len = 0
    for req in current_batch:
        sum_req_len += req.num_tokens

    # Iterate through possible iterations to find the peak memory
    mem_max = 0
    mem_max_iter = 0
    memory_usage_list = []
    for i in iter_list:
        memory = calculate_iter_mem(
            exp_gen_len_cache=exp_gen_len_cache,
            len_prob_cache=len_prob_cache,
            num_iter=i,
            sum_req_len=sum_req_len,
            current_batch=current_batch,
            length_thresholds=length_thresholds,
            expect_precision=expect_precision,
        )
        memory_usage_list.append(memory)
        if memory > mem_max:
            mem_max = memory
            mem_max_iter = i
    
    return memory_usage_list, mem_max, mem_max_iter