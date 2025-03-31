import vkdispatch as vd
import numpy as np
import dataclasses
from typing import List, Tuple

from .prime_utils import prime_factors, group_primes, DEFAULT_REGISTER_LIMIT

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

    def __init__(self, primes: List[int], max_register_count: int, N: int):
        self.primes = tuple(primes)
        self.fft_length = int(np.round(np.prod(primes)))
        self.instance_count = max_register_count // self.fft_length

        self.registers_used = self.fft_length * self.instance_count

        self.remainder = N % self.registers_used
        assert self.remainder % self.fft_length == 0, "Remainder must be divisible by the FFT length"
        self.remainder_offset = 1 if self.remainder != 0 else 0
        self.extra_ffts = self.remainder // self.fft_length

        self.thread_count = N // self.registers_used + self.remainder_offset

    def __str__(self):
        return f"FFT Stage Config:\n\tprimes: {self.primes}\n\t fft_length: {self.fft_length}\n\t instance_count: {self.instance_count}\n\tregisters_used: {self.registers_used}\n"
    
    def __repr__(self):
        return str(self)

@dataclasses.dataclass
class FFTParams:
    config: "FFTConfig" = None
    inverse: bool = False
    normalize: bool = True
    r2c: bool = False
    batch_y_stride: int = None
    batch_z_stride: int = None
    fft_stride: int = None
    angle_factor: float = None
    input_map: vd.MappingFunction = None
    input_buffers: List[vd.Buffer] = None
    output_map: vd.MappingFunction = None
    output_buffers: List[vd.Buffer] = None

@dataclasses.dataclass
class FFTConfig:
    N: int
    register_count: int
    stages: Tuple[FFTRegisterStageConfig]
    thread_counts: Tuple[int, int, int]
    fft_stride: int
    batch_y_stride: int
    batch_y_count: int
    batch_z_stride: int
    batch_z_count: int
    batch_threads: int
    exec_size: Tuple[int, int, int]

    def __init__(self, buffer_shape: Tuple, axis: int = None, max_register_count: int = None):
        if axis is None:
            axis = len(buffer_shape) - 1

        total_buffer_length = np.round(np.prod(buffer_shape)).astype(np.int32)

        N = buffer_shape[axis]

        self.fft_stride = np.round(np.prod(buffer_shape[axis + 1:])).astype(np.int32)
        self.batch_y_stride = self.fft_stride * N
        self.batch_y_count = total_buffer_length // self.batch_y_stride

        self.batch_z_stride = 1
        self.batch_z_count = self.fft_stride
        
        self.N = N

        if max_register_count is None:
            max_register_count = DEFAULT_REGISTER_LIMIT

        max_register_count = min(max_register_count, N)

        prime_groups = group_primes(prime_factors(N), max_register_count)
        self.stages = tuple([FFTRegisterStageConfig(group, max_register_count, N) for group in prime_groups])
        register_utilizations = [stage.registers_used for stage in self.stages]
        self.register_count = max(register_utilizations)


        self.thread_counts = [stage.thread_count for stage in self.stages]

        self.batch_threads = max(self.thread_counts)
        self.exec_size = (self.batch_threads, self.batch_y_count, self.batch_z_count)

    def __str__(self):
        return f"FFT Config:\nN: {self.N}\nregister_count: {self.register_count}\nstages:\n{self.stages}\nlocal_size: {self.thread_counts}"
    
    def __repr__(self):
        return str(self)
    
    def params(self,
               inverse: bool = False,
               normalize: bool = True,
               r2c: bool = False,
               input_map: vd.MappingFunction = None,
               input_buffers: List[vd.Buffer] = None,
               output_map: vd.MappingFunction = None,
               output_buffers: List[vd.Buffer] = None) -> FFTParams:
        return FFTParams(
            config=self,
            inverse=inverse,
            normalize=normalize,
            r2c=r2c,
            batch_y_stride=self.batch_y_stride,
            batch_z_stride=self.batch_z_stride,
            fft_stride=self.fft_stride,
            angle_factor=2 * np.pi * (1 if inverse else -1),
            input_map=input_map,
            input_buffers=input_buffers,
            output_map=output_map,
            output_buffers=output_buffers
        )
    
