import vkdispatch as vd
import vkdispatch.codegen as vc
import dataclasses
from typing import List, Tuple, Optional, Dict

from ..compat import numpy_compat as npc
import vkdispatch.base.dtype as dtypes
from .prime_utils import prime_factors, group_primes, default_register_limit, default_max_prime

from .stages import FFTRegisterStageConfig

def plan_fft_stages(N: int, max_register_count: int, compute_item_size: int) -> Tuple[FFTRegisterStageConfig]:
    all_factors = prime_factors(N)

    for factor in all_factors:
        if factor > default_max_prime():
            raise ValueError(f"A prime factor of {N} is {factor}, which exceeds the maximum prime supported {default_max_prime()}")

    prime_groups = group_primes(all_factors, max_register_count)

    stages = []
    input_stride = 1

    for group in prime_groups:
        stage = FFTRegisterStageConfig(
            group,
            max_register_count,
            N,
            compute_item_size,
            input_stride
        )
        stages.append(stage)
        input_stride = stage.output_stride

    return tuple(stages)

@dataclasses.dataclass
class FFTPlanCandidate:
    max_register_count: int
    stages: Tuple[FFTRegisterStageConfig]
    register_count: int
    batch_threads: int
    transfer_count: Optional[int] = None

    def __init__(self, N: int, max_register_count: int,compute_item_size: int):
        stages = plan_fft_stages(N, max_register_count, compute_item_size)
        register_count = max(stage.registers_used for stage in stages)
        batch_threads = max(stage.thread_count for stage in stages)

        if register_count > max_register_count:
            self.max_register_count = None
            self.stages = None
            self.register_count = None
            self.batch_threads = None
            self.transfer_count = None
            return

        transfer_count = 0
        output_stride = 1

        for stage_index in range(len(stages) - 1):
            output_stage = stages[stage_index]
            input_stage = stages[stage_index + 1]

            output_keys = output_stage.get_output_format(register_count).keys()
            input_keys = input_stage.get_input_format(register_count).keys()

            if output_keys != input_keys:
                transfer_count += 1

            output_stride *= output_stage.fft_length

        self.max_register_count = max_register_count
        self.stages = stages
        self.register_count = register_count
        self.batch_threads = batch_threads
        self.transfer_count = transfer_count

def register_limit_candidates(N: int, initial_limit: int) -> List[int]:
    divisors = {1}

    for factor in prime_factors(N):
        divisors.update(divisor * factor for divisor in tuple(divisors))

    candidates = [initial_limit]
    candidates.extend(
        divisor
        for divisor in sorted(divisors)
        if initial_limit < divisor <= N
    )
    return candidates

def required_batch_threads_limit(batch_inner_count: int) -> int:
    context = vd.get_context()
    thread_dimension_limit = (
        context.max_workgroup_size[1]
        if batch_inner_count > 1
        else context.max_workgroup_size[0]
    )
    return max(1, min(int(thread_dimension_limit), int(context.max_workgroup_invocations)))

def select_fft_plan_candidate(
    N: int,
    batch_inner_count: int,
    compute_item_size: int,
    max_register_count: Optional[int],
) -> FFTPlanCandidate:
    batch_threads_limit = required_batch_threads_limit(batch_inner_count)
    dimension_name = "y" if batch_inner_count > 1 else "x"

    if max_register_count is not None:
        requested_limit = min(max_register_count, N)
        candidate = FFTPlanCandidate(
            N=N,
            max_register_count=requested_limit,
            compute_item_size=compute_item_size,
        )

        if candidate.stages is None:
            raise ValueError(f"Failed to create an FFT plan candidate for N={N} with max_register_count={requested_limit}")

        if candidate.batch_threads <= batch_threads_limit:
            return candidate

        best_candidate = candidate
        explicit_text = "requested"
        searched_limit = requested_limit
    else:
        max_registers = default_register_limit()

        if N==16 or N==8 or N==4 or N==2 and vd.get_devices()[0].is_nvidia():
            max_registers = max(2, N//2)

        baseline_limit = min(8, N)
        requested_limit = baseline_limit
        candidate_limits = register_limit_candidates(max_registers, baseline_limit)
        searched_limit = candidate_limits[-1]

        baseline_candidate = FFTPlanCandidate(
            N=N,
            max_register_count=baseline_limit,
            compute_item_size=compute_item_size,
        )
        best_candidate = baseline_candidate if baseline_candidate.stages is not None else None

        if best_candidate is not None and baseline_candidate.batch_threads <= batch_threads_limit:
            for candidate_limit in candidate_limits[1:]:
                candidate = FFTPlanCandidate(
                    N=N,
                    max_register_count=candidate_limit,
                    compute_item_size=compute_item_size,
                )

                if candidate.stages is None:
                    continue

                if best_candidate is None or candidate.batch_threads < best_candidate.batch_threads:
                    best_candidate = candidate

                if candidate.batch_threads > batch_threads_limit:
                    continue

                if candidate.transfer_count < baseline_candidate.transfer_count:
                    return candidate

            return baseline_candidate

        for candidate_limit in candidate_limits[1:]:
            candidate = FFTPlanCandidate(
                N=N,
                max_register_count=candidate_limit,
                compute_item_size=compute_item_size,
            )
            if candidate.stages is None:
                continue

            if best_candidate is None or candidate.batch_threads < best_candidate.batch_threads:
                best_candidate = candidate

            if candidate.batch_threads <= batch_threads_limit:
                return candidate

        explicit_text = "default"

    raise ValueError(
        f"Unable to build an FFT plan for size {N}: minimum achievable batch thread count "
        f"{best_candidate.batch_threads} exceeds the device's local {dimension_name}-dimension "
        f"limit {batch_threads_limit} (starting from {explicit_text} max_register_count="
        f"{requested_limit}, searched up to {searched_limit})."
    )

@dataclasses.dataclass
class FFTConfig:
    N: int
    compute_type: dtypes.dtype
    register_count: int
    max_prime_radix: int
    stages: Tuple[FFTRegisterStageConfig]
    thread_counts: Tuple[int, int, int]
    fft_stride: int
    batch_outer_stride: int
    batch_outer_count: int
    batch_inner_count: int
    batch_threads: int
    sdata_allocation: int

    sdata_row_size: int
    sdata_row_size_padded: int

    def __init__(
        self,
        buffer_shape: Tuple,
        axis: int = None,
        max_register_count: int = None,
        compute_type: dtypes.dtype = vd.complex64,
    ):
        if axis is None:
            axis = len(buffer_shape) - 1

        if not dtypes.is_complex(compute_type):
            raise ValueError(f"compute_type must be a complex dtype, got {compute_type}")

        self.compute_type = compute_type

        total_buffer_length = int(round(npc.prod(buffer_shape)))

        N = buffer_shape[axis]

        self.fft_stride = int(round(npc.prod(buffer_shape[axis + 1:])))
        self.batch_outer_stride = self.fft_stride * N
        self.batch_outer_count = total_buffer_length // self.batch_outer_stride

        self.batch_inner_count = self.fft_stride
        
        self.N = N

        all_factors = prime_factors(N)

        for factor in all_factors:
            if factor > default_max_prime():
                raise ValueError(f"A prime factor of {N} is {factor}, which exceeds the maximum prime supported {default_max_prime()}")

        self.max_prime_radix = max(all_factors)

        plan_candidate = select_fft_plan_candidate(
            N=N,
            batch_inner_count=self.batch_inner_count,
            compute_item_size=self.compute_type.item_size,
            max_register_count=max_register_count,
        )
        self.stages = plan_candidate.stages
        self.register_count = plan_candidate.register_count

        self.sdata_allocation = 1
        self.sdata_row_size = 1
        self.sdata_row_size_padded = 1

        for stage in self.stages:
            if stage.sdata_size < self.sdata_allocation:
                continue

            self.sdata_allocation = stage.sdata_size
            self.sdata_row_size = stage.sdata_width
            self.sdata_row_size_padded = stage.sdata_width_padded

        self.thread_counts = tuple(stage.thread_count for stage in self.stages)

        self.batch_threads = plan_candidate.batch_threads

    def __str__(self):
        return f"FFT Config:\nN: {self.N}\nregister_count: {self.register_count}\nstages:\n{self.stages}\nlocal_size: {self.thread_counts}"
    
    def __repr__(self):
        return str(self)
    
    def angle_factor(self, inverse: bool) -> float:
        return 2 * npc.pi * (1 if inverse else -1)
