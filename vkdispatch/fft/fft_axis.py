import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import dataclasses
from typing import List, Tuple
from functools import lru_cache
import numpy as np

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
class FFTAxisConfig:
    N: int
    register_count: int
    stages: Tuple[FFTRegisterStageConfig]
    local_size: Tuple[int, int, int]

    def __init__(self, N: int, max_register_count: int = None):
        self.N = N

        if max_register_count is None:
            max_register_count = DEFAULT_REGISTER_LIMIT

        max_register_count = min(max_register_count, N)

        prime_groups = group_primes(prime_factors(N), max_register_count)
        self.stages = tuple([FFTRegisterStageConfig(group, max_register_count, N) for group in prime_groups])
        register_utilizations = [stage.registers_used for stage in self.stages]
        self.register_count = max(register_utilizations)

        thread_counts = [stage.thread_count for stage in self.stages]

        self.local_size = (max(thread_counts), 1, 1)

    def __str__(self):
        return f"FFT Axis Config:\nN: {self.N}\nregister_count: {self.register_count}\nstages:\n{self.stages}\nlocal_size: {self.local_size}"
    
    def __repr__(self):
        return str(self)

@dataclasses.dataclass
class FFTRegisterStageInvocation:
    stage: FFTRegisterStageConfig
    output_stride: int
    block_width: int
    inner_block_offset: int
    block_index: int
    sub_sequence_offset: int
    register_selection: slice

    def __init__(self, stage: FFTRegisterStageConfig, output_stride: int, instance_index: int, tid: vc.ShaderVariable):
        self.stage = stage
        self.output_stride = output_stride

        self.block_width = output_stride * stage.fft_length

        self.instance_id = tid * stage.instance_count + instance_index

        self.inner_block_offset = self.instance_id % output_stride

        if output_stride == 1:
            self.inner_block_offset = 0

        self.block_index = (self.instance_id * stage.fft_length) / self.block_width
        self.sub_sequence_offset = self.block_index * self.block_width + self.inner_block_offset

        self.register_selection = slice(instance_index * stage.fft_length, (instance_index + 1) * stage.fft_length)

class FFTAxisPlanner:
    def __init__(self, N: int, batch_y_stride: int = None, batch_z_stride: int = None, fft_stride: int = None, max_register_count: int = None, name: str = None):
        if name is None:
            name = f"fft_axis_{N}"

        self.config = FFTAxisConfig(N, max_register_count)

        #print(self.config)
        
        self.fft_stride = fft_stride if fft_stride is not None else 1

        self.builder = vc.ShaderBuilder(enable_exec_bounds=False)
        old_builder = vc.set_global_builder(self.builder)

        self.signature = vd.ShaderSignature.from_type_annotations(self.builder, [Buff[c64]])
        self.buffer = self.signature.get_variables()[0]

        self.batch_y_stride = batch_y_stride if batch_y_stride is not None else N
        self.batch_z_stride = batch_z_stride if batch_z_stride is not None else 1
        
        # Register allocation
        self.registers = [vc.new(c64, 0, var_name=f"register_{i}") for i in range(self.config.register_count)]
        self.radix_registers = [vc.new(c64, 0, var_name=f"radix_{i}") for i in range(self.config.register_count)]
        self.omega_register = vc.new(c64, 0, var_name="omega_register")

        # Local ID within the workgroup
        self.tid = vc.local_invocation().x.copy("tid")

        # Index offset of the current batch
        self.batch_offset = (vc.workgroup().y * self.batch_y_stride + vc.workgroup().z * self.batch_z_stride).copy("batch_offset")

        self.sdata = vc.shared_buffer(vc.c64, self.config.N, "sdata")

        self.plan()

        vc.set_global_builder(old_builder)

        self.description = self.builder.build(name)

    def load_buffer_to_registers(self, buffer: Buff[c64], offset: Const[u32], stride: Const[u32], register_list: List[vc.ShaderVariable] = None):
        if register_list is None:
            register_list = self.registers

        batch_offset = self.batch_offset
        fft_stride = self.fft_stride

        if buffer is None:
            buffer = self.sdata
            batch_offset = 0
            fft_stride = 1

        vc.comment(f"Loading to registers {register_list} from buffer {buffer} at offset {offset} and stride {stride}")

        for i in range(len(register_list)):
            register_list[i][:] = buffer[(i * stride + offset) * fft_stride  + batch_offset]

    def store_registers_in_buffer(self, buffer: Buff[c64], offset: Const[u32], stride: Const[u32], register_list: List[vc.ShaderVariable] = None):
        if register_list is None:
            register_list = self.registers

        batch_offset = self.batch_offset
        fft_stride = self.fft_stride

        if buffer is None:
            buffer = self.sdata
            batch_offset = 0
            fft_stride = 1

        vc.comment(f"Storing registers {register_list} to buffer {buffer} at offset {offset} and stride {stride}")

        for i in range(len(register_list)):
            buffer[(i * stride + offset) * fft_stride + batch_offset] = register_list[i]
    
    def radix_P(self, register_list: List[vc.ShaderVariable]):
        assert len(register_list) <= len(self.radix_registers), "Too many registers for radix_P"

        if len(register_list) == 1:
            return

        vc.comment(f"Performing a DFT for Radix-{len(register_list)} FFT")

        for i in range(0, len(register_list)):
            for j in range(0, len(register_list)):
                if j == 0:
                    self.radix_registers[i][:] = register_list[j]
                    continue

                if i == 0:
                    self.radix_registers[i] += register_list[j]
                    continue

                if i * j == len(register_list) // 2 and len(register_list) % 2 == 0:
                    self.radix_registers[i] -= register_list[j]
                    continue

                omega = np.exp(-2j * np.pi * i * j / len(register_list))
                self.radix_registers[i] += vc.mult_c64_by_const(register_list[j], omega)

        for i in range(0, len(register_list)):
            register_list[i][:] = self.radix_registers[i]

    def apply_cooley_tukey_twiddle_factors(self, register_list: List[vc.ShaderVariable], twiddle_index: int = 0, twiddle_N: int = 1):
        if isinstance(twiddle_index, int) and twiddle_index == 0:
            return

        vc.comment(f"Applying Cooley-Tukey twiddle factors for twiddle index {twiddle_index} and twiddle N {twiddle_N}")

        for i in range(len(register_list)):
            if i == 0:
                continue
            
            if isinstance(twiddle_index, int):
                if twiddle_index == 0:
                    #register_list[i][:] = vc.mult_c64_by_const(register_list[i], omega)
                    continue

                omega = np.exp( -2j * np.pi * i * twiddle_index / twiddle_N)
                register_list[i][:] = vc.mult_c64_by_const(register_list[i], omega)
                continue
            
            self.omega_register.x = -2 * np.pi * i * twiddle_index / twiddle_N
            self.omega_register[:] = vc.complex_from_euler_angle(self.omega_register.x)
            self.omega_register[:] = vc.mult_c64(self.omega_register, register_list[i])

            register_list[i][:] = self.omega_register

    def register_radix_composite(self, register_list: List[vc.ShaderVariable], primes: List[int]):
        if len(register_list) == 1:
            return
        
        N = len(register_list)

        assert N == np.prod(primes), "Product of primes must be equal to the number of registers"

        vc.comment(f"Performing a Radix-{primes} FFT on {N} registers")

        output_stride = 1

        for prime in primes:
            sub_squences = [register_list[i::N//prime] for i in range(N//prime)]

            block_width = output_stride * prime

            for i in range(0, N // prime):
                inner_block_offset = i % output_stride
                block_index = (i * prime) // block_width

                self.apply_cooley_tukey_twiddle_factors(sub_squences[i], twiddle_index=inner_block_offset, twiddle_N=block_width)
                self.radix_P(sub_squences[i])
                
                sub_sequence_offset = block_index * block_width + inner_block_offset

                for j in range(prime):
                    register_list[sub_sequence_offset + j * output_stride] = sub_squences[i][j]
            
            output_stride *= prime   

        return register_list

    def process_fft_register_stage(self, stage: FFTRegisterStageConfig, output_stride: int, input = None, output = None):
        do_runtime_if = stage.thread_count < self.config.local_size[0]
        
        vc.comment(f"Processing prime group {stage.primes} by doing {stage.instance_count} radix-{stage.fft_length} FFTs on {self.config.N // stage.registers_used} groups")
        if do_runtime_if: vc.if_statement(self.tid < stage.thread_count)

        stage_invocations: List[FFTRegisterStageInvocation] = []

        for i in range(stage.instance_count):
            stage_invocations.append(FFTRegisterStageInvocation(stage, output_stride, i , self.tid))

        for ii, invocation in enumerate(stage_invocations):
            if stage.remainder_offset == 1 and ii == stage.extra_ffts:
                vc.if_statement(self.tid < self.config.N // stage.registers_used)

            self.load_buffer_to_registers(
                buffer=input, 
                offset=invocation.instance_id, 
                stride=self.config.N // stage.fft_length, 
                register_list=self.registers[invocation.register_selection]
            )

        if stage.remainder_offset == 1:
            vc.end()

        if do_runtime_if: vc.end()

        if input is None and output is None:
            vc.memory_barrier()
            vc.barrier()

        if do_runtime_if: vc.if_statement(self.tid < stage.thread_count)

        for ii, invocation in enumerate(stage_invocations):
            if stage.remainder_offset == 1 and ii == stage.extra_ffts:
                vc.if_statement(self.tid < self.config.N // stage.registers_used)

            self.apply_cooley_tukey_twiddle_factors(
                register_list=self.registers[invocation.register_selection], 
                twiddle_index=invocation.inner_block_offset, 
                twiddle_N=invocation.block_width
            )

            self.registers[invocation.register_selection] = self.register_radix_composite(
                register_list=self.registers[invocation.register_selection],
                primes=stage.primes
            )

            self.store_registers_in_buffer(
                buffer=output,
                offset=invocation.sub_sequence_offset,
                stride=output_stride,
                register_list=self.registers[invocation.register_selection]
            )
        
        if stage.remainder_offset == 1:
            vc.end()

        if do_runtime_if: vc.end()

    def plan(self):
        output_stride = 1

        stage_count = len(self.config.stages)

        for i in range(stage_count):
            self.process_fft_register_stage(
                self.config.stages[i],
                output_stride,
                input=self.buffer if i == 0 else None,
                output=self.buffer if i == stage_count - 1 else None)
            
            output_stride *= self.config.stages[i].fft_length

            if i < stage_count - 1:
                vc.memory_barrier()
                vc.barrier()

@lru_cache(maxsize=None)
def make_fft_stage(
        N: int, 
        stride: int = 1,
        batch_y_stride: int = None,
        batch_z_stride: int = None,
        name: str = None,
        max_register_count: int = None) -> vd.ShaderObject:

    axis_planner = FFTAxisPlanner(
        N, 
        name=name, 
        batch_y_stride=batch_y_stride,
        batch_z_stride=batch_z_stride,
        fft_stride=stride,
        max_register_count=max_register_count)

    return vd.ShaderObject(axis_planner.description, axis_planner.signature, local_size=axis_planner.config.local_size)

def get_cache_info():
    return make_fft_stage.cache_info()

def print_cache_info():
    print(get_cache_info())

def cache_clear():
    return make_fft_stage.cache_clear()