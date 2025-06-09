import vkdispatch as vd
import numpy as np
import dataclasses
from typing import List, Tuple, Optional

from .prime_utils import prime_factors, group_primes, default_register_limit, default_max_prime

@dataclasses.dataclass
class FFTRegisterStageConfig:
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

    def __init__(self, primes: List[int], max_register_count: int, N: int):
        self.primes = tuple(primes)
        self.fft_length = int(np.round(np.prod(primes)))
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

        self.sdata_size = self.sdata_width_padded * int(np.prod(threads_primes))

        if self.sdata_size > vd.get_context().max_shared_memory // vd.complex64.item_size:
            self.sdata_width_padded = self.sdata_width
            self.sdata_size = self.sdata_width_padded * int(np.prod(threads_primes))

    def __str__(self):
        return f"""
FFT Stage Config:
    primes: {self.primes}
    fft_length: {self.fft_length}
    instance_count: {self.instance_count}
    registers_used: {self.registers_used}
    remainder: {self.remainder}
    remainder_offset: {self.remainder_offset}
    extra_ffts: {self.extra_ffts}
    thread_count: {self.thread_count}
    sdata_size: {self.sdata_size}
    sdata_width: {self.sdata_width}
    sdata_width_padded: {self.sdata_width_padded}"""
    
    def __repr__(self):
        return str(self)

@dataclasses.dataclass
class FFTParams:
    config: "FFTConfig" = None
    inverse: bool = False
    normalize: bool = True
    r2c: bool = False
    batch_outer_stride: int = None
    batch_inner_stride: int = None
    fft_stride: int = None
    angle_factor: float = None
    input_sdata: bool = False
    input_buffers: List[vd.Buffer] = None
    output_buffers: List[vd.Buffer] = None
    passthrough: bool = False

    sdata_row_size: Optional[int] = None
    sdata_row_size_padded: Optional[int] = None


@dataclasses.dataclass
class FFTConfig:
    N: int
    register_count: int
    max_prime_radix: int
    stages: Tuple[FFTRegisterStageConfig]
    thread_counts: Tuple[int, int, int]
    fft_stride: int
    batch_outer_stride: int
    batch_outer_count: int
    batch_inner_stride: int
    batch_inner_count: int
    batch_threads: int
    sdata_allocation: int

    sdata_row_size: Optional[int]
    sdata_row_size_padded: Optional[int]

    def __init__(self, buffer_shape: Tuple, axis: int = None, max_register_count: int = None):
        if axis is None:
            axis = len(buffer_shape) - 1

        total_buffer_length = np.round(np.prod(buffer_shape)).astype(np.int32)

        N = buffer_shape[axis]

        self.fft_stride = np.round(np.prod(buffer_shape[axis + 1:])).astype(np.int32)
        self.batch_outer_stride = self.fft_stride * N
        self.batch_outer_count = total_buffer_length // self.batch_outer_stride

        self.batch_inner_stride = 1
        self.batch_inner_count = self.fft_stride
        
        self.N = N

        if max_register_count is None:
            max_register_count = default_register_limit()

        max_register_count = min(max_register_count, N)

        all_factors = prime_factors(N)

        for factor in all_factors:
            assert factor <= default_max_prime(), f"A prime factor of {N} is {factor}, which exceeds the maximum prime supported {default_max_prime()}"

        self.max_prime_radix = max(all_factors)

        prime_groups = group_primes(all_factors, max_register_count)        

        self.stages = tuple([FFTRegisterStageConfig(group, max_register_count, N) for group in prime_groups])
        register_utilizations = [stage.registers_used for stage in self.stages]
        self.register_count = max(register_utilizations)

        assert self.register_count <= max_register_count, f"Register count {self.register_count} exceeds max register count {max_register_count}"

        self.sdata_allocation = 1 

        for stage in self.stages:
            if stage.sdata_size < self.sdata_allocation:
                continue

            self.sdata_allocation = stage.sdata_size
            self.sdata_row_size = stage.sdata_width
            self.sdata_row_size_padded = stage.sdata_width_padded

        self.thread_counts = [stage.thread_count for stage in self.stages]

        self.batch_threads = max(self.thread_counts)

    def __str__(self):
        return f"FFT Config:\nN: {self.N}\nregister_count: {self.register_count}\nstages:\n{self.stages}\nlocal_size: {self.thread_counts}"
    
    def __repr__(self):
        return str(self)
    
    def params(self,
               inverse: bool = False,
               normalize: bool = True,
               r2c: bool = False,
               input_sdata: bool = False,
               input_buffers: List[vd.Buffer] = None,
               output_buffers: List[vd.Buffer] = None,
               passthrough: bool = False) -> FFTParams:
        return FFTParams(
            config=self,
            inverse=inverse,
            normalize=normalize,
            r2c=r2c,
            batch_outer_stride=self.batch_outer_stride,
            batch_inner_stride=self.batch_inner_stride,
            fft_stride=self.fft_stride,
            angle_factor=2 * np.pi * (1 if inverse else -1),
            input_sdata=input_sdata,
            input_buffers=input_buffers,
            output_buffers=output_buffers,
            passthrough=passthrough,
            sdata_row_size=self.sdata_row_size,
            sdata_row_size_padded=self.sdata_row_size_padded
        )
    
