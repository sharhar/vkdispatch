import vkdispatch as vd
import vkdispatch.codegen as vc
import dataclasses
from typing import List, Tuple, Optional

from ..compat import numpy_compat as npc
import vkdispatch.base.dtype as dtypes
from .prime_utils import prime_factors, group_primes, default_register_limit, default_max_prime


@dataclasses.dataclass(frozen=True)
class _FFTPlanCandidate:
    max_register_count: int
    stages: Tuple["FFTRegisterStageConfig", ...]
    register_count: int
    batch_threads: int


def _default_max_register_count(N: int) -> int:
    max_register_count = default_register_limit()

    if N == 16 or N == 8 or N == 4 or (N == 2 and vd.get_devices()[0].is_nvidia()):
        max_register_count = max(2, N // 2)

    return min(max_register_count, N)


def _required_batch_threads_limit(batch_inner_count: int) -> int:
    context = vd.get_context()
    thread_dimension_limit = (
        context.max_workgroup_size[1]
        if batch_inner_count > 1
        else context.max_workgroup_size[0]
    )
    return max(1, min(int(thread_dimension_limit), int(context.max_workgroup_invocations)))


def _evaluate_fft_plan_candidate(
    N: int,
    all_factors: List[int],
    max_register_count: int,
    compute_item_size: int,
) -> _FFTPlanCandidate:
    prime_groups = group_primes(all_factors, max_register_count)
    stages = tuple(
        FFTRegisterStageConfig(group, max_register_count, N, compute_item_size)
        for group in prime_groups
    )
    register_count = max(stage.registers_used for stage in stages)
    batch_threads = max(stage.thread_count for stage in stages)

    assert register_count <= max_register_count, (
        f"Register count {register_count} exceeds max register count {max_register_count}"
    )

    return _FFTPlanCandidate(
        max_register_count=max_register_count,
        stages=stages,
        register_count=register_count,
        batch_threads=batch_threads,
    )


def _register_limit_candidates(N: int, initial_limit: int) -> List[int]:
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


def _select_fft_plan_candidate(
    N: int,
    all_factors: List[int],
    batch_inner_count: int,
    compute_item_size: int,
    max_register_count: Optional[int],
) -> _FFTPlanCandidate:
    batch_threads_limit = _required_batch_threads_limit(batch_inner_count)
    dimension_name = "y" if batch_inner_count > 1 else "x"

    if max_register_count is None:
        requested_limit = _default_max_register_count(N)
        candidate_limits = _register_limit_candidates(N, requested_limit)
        searched_limit = candidate_limits[-1]
        explicit_limit = False
    else:
        requested_limit = min(max_register_count, N)
        candidate_limits = [requested_limit]
        searched_limit = requested_limit
        explicit_limit = True

    best_candidate = None

    for candidate_limit in candidate_limits:
        candidate = _evaluate_fft_plan_candidate(
            N=N,
            all_factors=all_factors,
            max_register_count=candidate_limit,
            compute_item_size=compute_item_size,
        )
        if best_candidate is None or candidate.batch_threads < best_candidate.batch_threads:
            best_candidate = candidate

        if candidate.batch_threads <= batch_threads_limit:
            return candidate

    explicit_text = "requested" if explicit_limit else "default"
    raise ValueError(
        f"Unable to build an FFT plan for size {N}: minimum achievable batch thread count "
        f"{best_candidate.batch_threads} exceeds the device's local {dimension_name}-dimension "
        f"limit {batch_threads_limit} (starting from {explicit_text} max_register_count="
        f"{requested_limit}, searched up to {searched_limit})."
    )

@dataclasses.dataclass
class FFTRegisterStageConfig:
    """
    Configuration for an FFT register stage.

    Attributes:

        primes (Tuple[int]): The prime numbers used for factorization.
        fft_length (int): The length of each FFT stage.
        instance_count (int): The number of instances required to achieve the desired level of parallelism.
        registers_used (int): The total number of registers used by the FFT stage.
        remainder (int): The remainder of `N` divided by `registers_used`.
        remainder_offset (int): A flag indicating whether the remainder is non-zero.
        extra_ffts (int): The additional number of FFT stages required to process the remainder.
        thread_count (int): The total number of threads used in the computation.
        sdata_size (int): The size of the shared memory buffer used to store intermediate results.
        sdata_width (int): The width of each element in the shared memory buffer.
        sdata_width_padded (int): The padded width of each element in the shared memory buffer.

    """

    primes: Tuple[int]
    fft_length: int
    instance_count: int
    registers_used: int
    remainder: int
    remainder_offset: int
    extra_ffts: int
    thread_count: int
    sdata_size: int
    sdata_width: int
    sdata_width_padded: int

    def __init__(self, primes: List[int], max_register_count: int, N: int, compute_item_size: int):
        """
        Initializes the FFTRegisterStageConfig object.

        Parameters:

            primes (List[int]): The prime numbers to use for factorization.
            max_register_count (int): The maximum number of registers allowed per thread.
            N (int): The length of the input data.

        """
        self.primes = tuple(primes)
        self.fft_length = int(round(npc.prod(primes)))
        instance_primes = prime_factors(N // self.fft_length)
 
        self.instance_count = 1

        while len(instance_primes) > 0:
            if self.instance_count * self.fft_length * instance_primes[0] > max_register_count:
                break
            self.instance_count *= instance_primes[0]
            instance_primes = instance_primes[1:]

        self.registers_used = self.fft_length * self.instance_count

        self.remainder = N % self.registers_used
        assert self.remainder % self.fft_length == 0, "Remainder must be divisible by the FFT length"
        self.remainder_offset = 1 if self.remainder != 0 else 0
        self.extra_ffts = self.remainder // self.fft_length

        self.thread_count = N // self.registers_used + self.remainder_offset

        self.sdata_width = self.registers_used

        threads_primes = prime_factors(self.thread_count)

        while self.sdata_width < 16 and len(threads_primes) > 0:
            self.sdata_width *= threads_primes[0]
            threads_primes = threads_primes[1:]

        self.sdata_width_padded = self.sdata_width

        if self.sdata_width_padded % 2 == 0:
            self.sdata_width_padded += 1

        self.sdata_size = self.sdata_width_padded * int(npc.prod(threads_primes))

        if self.sdata_size > vd.get_context().max_shared_memory // compute_item_size:
            self.sdata_width_padded = self.sdata_width
            self.sdata_size = self.sdata_width_padded * int(npc.prod(threads_primes))

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
            assert factor <= default_max_prime(), f"A prime factor of {N} is {factor}, which exceeds the maximum prime supported {default_max_prime()}"

        self.max_prime_radix = max(all_factors)

        plan_candidate = _select_fft_plan_candidate(
            N=N,
            all_factors=all_factors,
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
