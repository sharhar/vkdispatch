import vkdispatch as vd
import vkdispatch.codegen as vc
import dataclasses
from typing import List, Tuple, Dict

from ..compat import numpy_compat as npc
from .prime_utils import prime_factors

@dataclasses.dataclass
class FFTStagePlanInvocation:
    fft_length: int
    input_stride: int
    instance_index: int
    instance_index_stride: int
    block_width: int
    full_width_block: bool
    instance_id0: int
    inner_block_offset0: int
    sub_sequence_offset0: int
    register_selection: slice

    def __init__(self,
                 stage_fft_length: int,
                 stage_instance_count: int,
                 input_stride: int,
                 instance_index: int,
                 N: int):
        self.fft_length = stage_fft_length
        self.input_stride = input_stride
        self.instance_index = instance_index
        self.block_width = input_stride * stage_fft_length
        self.instance_index_stride = N // (stage_fft_length * stage_instance_count)

        self.full_width_block = self.block_width == N

        # pretend tid is 0, used for calculating register shuffles
        self.instance_id0 = self.instance_index_stride * instance_index
        self.inner_block_offset0 = self.instance_id0 % input_stride
        self.sub_sequence_offset0 = self.instance_id0 * stage_fft_length - self.inner_block_offset0 * (stage_fft_length - 1)
        
        self.register_selection = slice(instance_index * stage_fft_length, (instance_index + 1) * stage_fft_length)

    def get_offset(self, tid: vc.ShaderVariable):
        return tid + self.instance_index_stride * self.instance_index

    def get_inner_block_offset(self, tid: vc.ShaderVariable):
        if self.input_stride == 1:
            return 0

        if self.full_width_block:
            return self.get_offset(tid)

        return self.get_offset(tid) % self.input_stride

    def get_sub_sequence_offset(self, tid: vc.ShaderVariable):
        if self.full_width_block:
            return self.get_offset(tid)

        return self.get_offset(tid) * self.fft_length - self.get_inner_block_offset(tid) * (self.fft_length - 1)

    def get_write_index(self, fft_index: int):
        return self.sub_sequence_offset0 + fft_index * self.input_stride
    
    def get_read_index(self, offset: int):
        return self.instance_id0 + offset

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

    N: int
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
    input_stride: int
    output_stride: int
    invocations: Tuple[FFTStagePlanInvocation]

    def __init__(self, primes: List[int],
                 max_register_count: int,
                 N: int,
                 compute_item_size: int,
                 input_stride: int):
        """
        Initializes the FFTRegisterStageConfig object.

        Parameters:

            primes (List[int]): The prime numbers to use for factorization.
            max_register_count (int): The maximum number of registers allowed per thread.
            N (int): The length of the input data.

        """
        self.N = N
        self.primes = tuple(primes)
        self.input_stride = input_stride
        self.fft_length = int(round(npc.prod(primes)))
        self.output_stride = self.input_stride * self.fft_length
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

        invocations = []
        for instance_index in range(self.instance_count):
            invocations.append(FFTStagePlanInvocation(
                stage_fft_length=self.fft_length,
                stage_instance_count=self.instance_count,
                input_stride=input_stride,
                instance_index=instance_index,
                N=N
            ))

        self.invocations = tuple(invocations)

    def get_input_format(self, register_count: int) -> Dict[int, int]:
        in_format = {}

        stride = self.N // self.fft_length

        register_index_list = list(range(register_count))

        for invocation in self.invocations:
            sub_registers = register_index_list[invocation.register_selection]
            
            for i in range(len(sub_registers)):
                in_format[invocation.get_read_index(stride * i)] = sub_registers[i]

        return in_format

    def get_output_format(self, register_count: int) -> Dict[int, int]:
        out_format = {}

        register_index_list = list(range(register_count))

        for jj in range(self.fft_length):
            for invocation in self.invocations:
                out_format[invocation.get_write_index(jj)] = register_index_list[invocation.register_selection][jj]

        return out_format