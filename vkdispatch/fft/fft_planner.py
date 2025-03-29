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
    thread_counts: Tuple[int, int, int]

    def __init__(self, N: int, max_register_count: int = None):
        self.N = N

        if max_register_count is None:
            max_register_count = DEFAULT_REGISTER_LIMIT

        max_register_count = min(max_register_count, N)

        prime_groups = group_primes(prime_factors(N), max_register_count)
        self.stages = tuple([FFTRegisterStageConfig(group, max_register_count, N) for group in prime_groups])
        register_utilizations = [stage.registers_used for stage in self.stages]
        self.register_count = max(register_utilizations)

        self.thread_counts = [stage.thread_count for stage in self.stages]

    def __str__(self):
        return f"FFT Axis Config:\nN: {self.N}\nregister_count: {self.register_count}\nstages:\n{self.stages}\nlocal_size: {self.thread_counts}"
    
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

def allocation_valid(workgroup_size: int, shared_memory: int):
    return workgroup_size <= vd.get_context().max_workgroup_invocations and shared_memory <= vd.get_context().max_shared_memory

def allocate_inline_batches(batch_num: int, batch_threads: int, N: int, max_workgroup_size: int):
    batch_num_primes = prime_factors(batch_num)

    prime_index = len(batch_num_primes) - 1

    workgroup_size = batch_threads
    shared_memory_allocation = batch_threads * N * vd.complex64.item_size
    inline_batches = 1

    while allocation_valid(workgroup_size, shared_memory_allocation) and prime_index >= 0 and inline_batches <= max_workgroup_size:
        test_prime = batch_num_primes[prime_index]

        if allocation_valid(workgroup_size * test_prime, shared_memory_allocation * test_prime) and inline_batches * test_prime <= max_workgroup_size:
            workgroup_size *= test_prime
            shared_memory_allocation *= test_prime
            inline_batches *= test_prime
        
        prime_index -= 1

    return inline_batches

@dataclasses.dataclass
class FFTResources:
    registers: List[Const[c64]]
    radix_registers: List[Const[c64]]
    omega_register: Const[c64]
    tid: Const[u32]
    input_batch_offset: Const[u32]
    output_batch_offset: Const[u32]
    subsequence_offset: Const[u32]
    sdata: Buff[c64]
    sdata_offset: Const[u32]
    io_index: Const[u32]

    inline_batch_y: int
    inline_batch_z: int
    shared_memory_size: int
    local_size: Tuple[int, int, int]

    def reset(self):
        for register in self.registers:
            register[:] = "vec2(0)"
        
        for register in self.radix_registers:
            register[:] = "vec2(0)"
        
        self.omega_register[:] = "vec2(0)"

class FFTPlanner:
    def __init__(self, N: int, batch_y_stride: int = None, batch_z_stride: int = None, fft_stride: int = None, max_register_count: int = None, name: str = None):
        self.name = f"fft_axis_{N}" if name is None else name

        self.config = FFTAxisConfig(N, max_register_count)
        
        self.fft_stride = fft_stride if fft_stride is not None else 1
        self.batch_y_stride = batch_y_stride if batch_y_stride is not None else N
        self.batch_z_stride = batch_z_stride if batch_z_stride is not None else 1

        self.batch_threads = max(self.config.thread_counts)

        self.reset()

    def reset(self):
        self.buffer = None
        self.angle_factor = None
        self.normalize = None
        self.resources = None
        self.r2c = None
    
    def allocate_resources(self, batch_num_y: int = 1, batch_num_z: int = 1, r2c: bool = False) -> FFTResources:
        assert self.resources is None, "Resources already allocated"

        inline_batch_z = allocate_inline_batches(batch_num_z, self.batch_threads, self.config.N, vd.get_context().max_workgroup_size[2])
        inline_batch_y = allocate_inline_batches(batch_num_y, self.batch_threads * inline_batch_z, self.config.N, vd.get_context().max_workgroup_size[1])

        input_batch_stride_y = self.batch_y_stride
        output_batch_stride_y = self.batch_y_stride

        if r2c:
            output_batch_stride_y = (self.config.N // 2) + 1
            input_batch_stride_y = output_batch_stride_y * 2

        self.resources = FFTResources(
            registers=[vc.new(c64, 0, var_name=f"register_{i}") for i in range(self.config.register_count)],
            radix_registers=[vc.new(c64, 0, var_name=f"radix_{i}") for i in range(self.config.register_count)],
            omega_register=vc.new(c64, 0, var_name="omega_register"),
            tid=vc.local_invocation().x.copy("tid"),
            input_batch_offset=(vc.global_invocation().y * input_batch_stride_y + vc.global_invocation().z * self.batch_z_stride).copy("input_batch_offset"),
            output_batch_offset=(vc.global_invocation().y * output_batch_stride_y + vc.global_invocation().z * self.batch_z_stride).copy("output_batch_offset"),
            subsequence_offset=vc.new_uint(0, var_name="subsequence_offset"),
            sdata=vc.shared_buffer(vc.c64, self.config.N * inline_batch_y * inline_batch_z, "sdata"),
            sdata_offset=(vc.local_invocation().y * inline_batch_z * self.config.N + vc.local_invocation().z * self.config.N).copy("sdata_offset"),
            io_index=vc.new_uint(0, var_name="io_index"),
            inline_batch_y=inline_batch_y,
            inline_batch_z=inline_batch_z,
            shared_memory_size=self.config.N * inline_batch_y * inline_batch_z * vd.complex64.item_size,
            local_size=(self.batch_threads, inline_batch_y, inline_batch_z)
        )

    def release_resources(self):
        assert self.resources is not None, "Resources not allocated"

        self.resources = None

    def cache_info(self):
        return self.shader.cache_info()
    
    def cache_clear(self):
        return self.shader.cache_clear()

    def get_global_input(self, buffer: Buff, index: Const[u32]):
        self.resources.io_index[:] = index * self.fft_stride + self.resources.input_batch_offset

        if not self.r2c or self.angle_factor > 0:
            return buffer[self.resources.io_index]
        
        real_value = buffer[self.resources.io_index / 2][self.resources.io_index % 2]

        return f"vec2({real_value}, 0)"

    def load_buffer_to_registers(self, buffer: Buff, offset: Const[u32], stride: Const[u32], register_list: List[vc.ShaderVariable] = None):
        if register_list is None:
            register_list = self.resources.registers

        vc.comment(f"Loading to registers {register_list} from buffer {buffer} at offset {offset} and stride {stride}")

        for i in range(len(register_list)):
            register_list[i][:] = (
                self.get_global_input(buffer, i * stride + offset)
                if buffer is not None else
                self.resources.sdata[i * stride + offset + self.resources.sdata_offset]
            )

    def set_global_output(self, buffer: Buff, index: Const[u32], value: Const[c64]):
        if self.r2c and self.angle_factor < 0:
            vc.if_statement(
                index < (self.config.N // 2) + 1,
                f"{buffer[index * self.fft_stride + self.resources.output_batch_offset]} = {value};")
            
            return

        if self.angle_factor > 0 and self.normalize:
            buffer[index * self.fft_stride + self.resources.output_batch_offset] = value / self.config.N
        else:
            buffer[index * self.fft_stride + self.resources.output_batch_offset] = value

    def store_registers_in_buffer(self, buffer: Buff, offset: Const[u32], stride: Const[u32], register_list: List[vc.ShaderVariable] = None):
        if register_list is None:
            register_list = self.resources.registers

        vc.comment(f"Storing registers {register_list} to buffer {buffer} at offset {offset} and stride {stride}")

        for i in range(len(register_list)):
            if buffer is None:
                self.resources.sdata[i * stride + offset + self.resources.sdata_offset] = register_list[i]
            else:
                self.set_global_output(buffer, i * stride + offset, register_list[i])

            #buffer[(i * stride + offset) * self.fft_stride + self.resources.batch_offset] = write_object
    
    def radix_P(self, register_list: List[vc.ShaderVariable]):
        assert len(register_list) <= len(self.resources.radix_registers), "Too many registers for radix_P"

        if len(register_list) == 1:
            return

        vc.comment(f"Performing a DFT for Radix-{len(register_list)} FFT")

        for i in range(0, len(register_list)):
            for j in range(0, len(register_list)):
                if j == 0:
                    self.resources.radix_registers[i][:] = register_list[j]
                    continue

                if i == 0:
                    self.resources.radix_registers[i] += register_list[j]
                    continue

                if i * j == len(register_list) // 2 and len(register_list) % 2 == 0:
                    self.resources.radix_registers[i] -= register_list[j]
                    continue

                omega = np.exp(1j * self.angle_factor * i * j / len(register_list))
                self.resources.radix_registers[i] += vc.mult_c64_by_const(register_list[j], omega)

        for i in range(0, len(register_list)):
            register_list[i][:] = self.resources.radix_registers[i]

    def apply_cooley_tukey_twiddle_factors(self, register_list: List[vc.ShaderVariable], twiddle_index: int = 0, twiddle_N: int = 1):
        if isinstance(twiddle_index, int) and twiddle_index == 0:
            return

        vc.comment(f"Applying Cooley-Tukey twiddle factors for twiddle index {twiddle_index} and twiddle N {twiddle_N}")

        for i in range(len(register_list)):
            if i == 0:
                continue
            
            if isinstance(twiddle_index, int):
                if twiddle_index == 0:
                    continue

                omega = np.exp(1j * self.angle_factor * i * twiddle_index / twiddle_N)
                register_list[i][:] = vc.mult_c64_by_const(register_list[i], omega)
                continue
            
            self.resources.omega_register.x = self.angle_factor * i * twiddle_index / twiddle_N
            self.resources.omega_register[:] = vc.complex_from_euler_angle(self.resources.omega_register.x)
            self.resources.omega_register[:] = vc.mult_c64(self.resources.omega_register, register_list[i])

            register_list[i][:] = self.resources.omega_register

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

    def process_fft_register_stage(self, 
                                   stage: FFTRegisterStageConfig, 
                                   output_stride: int, 
                                   input = None, 
                                   output = None):
        do_runtime_if = stage.thread_count < self.batch_threads
        
        vc.comment(f"Processing prime group {stage.primes} by doing {stage.instance_count} radix-{stage.fft_length} FFTs on {self.config.N // stage.registers_used} groups")
        if do_runtime_if: vc.if_statement(self.resources.tid < stage.thread_count)

        stage_invocations: List[FFTRegisterStageInvocation] = []

        for i in range(stage.instance_count):
            stage_invocations.append(FFTRegisterStageInvocation(stage, output_stride, i , self.resources.tid))

        for ii, invocation in enumerate(stage_invocations):
            if stage.remainder_offset == 1 and ii == stage.extra_ffts:
                vc.if_statement(self.resources.tid < self.config.N // stage.registers_used)

            self.load_buffer_to_registers(
                buffer=input, 
                offset=invocation.instance_id, 
                stride=self.config.N // stage.fft_length, 
                register_list=self.resources.registers[invocation.register_selection]
            )

        if stage.remainder_offset == 1:
            vc.end()

        if do_runtime_if: vc.end()

        if input is None and output is None:
            vc.memory_barrier()
            vc.barrier()

        if do_runtime_if: vc.if_statement(self.resources.tid < stage.thread_count)

        for ii, invocation in enumerate(stage_invocations):

            if stage.remainder_offset == 1 and ii == stage.extra_ffts:
                vc.if_statement(self.resources.tid < self.config.N // stage.registers_used)

            self.apply_cooley_tukey_twiddle_factors(
                register_list=self.resources.registers[invocation.register_selection], 
                twiddle_index=invocation.inner_block_offset, 
                twiddle_N=invocation.block_width
            )

            self.resources.registers[invocation.register_selection] = self.register_radix_composite(
                register_list=self.resources.registers[invocation.register_selection],
                primes=stage.primes
            )
            
            self.resources.subsequence_offset[:] = invocation.sub_sequence_offset

            self.store_registers_in_buffer(
                buffer=output,
                offset=self.resources.subsequence_offset,
                stride=output_stride,
                register_list=self.resources.registers[invocation.register_selection]
            )
        
        if stage.remainder_offset == 1:
            vc.end()

        if do_runtime_if: vc.end()

    def plan(self, input: Buff = None, output: Buff = None, inverse: bool = False, normalize_inverse: bool = True, r2c: bool = False):
        self.angle_factor = 2 * np.pi * (1 if inverse else -1)
        self.normalize = normalize_inverse
        self.r2c = r2c

        output_stride = 1

        stage_count = len(self.config.stages)

        for i in range(stage_count):
            self.process_fft_register_stage(
                self.config.stages[i],
                output_stride,
                input=input if i == 0 else None,
                output=output if i == stage_count - 1 else None)
            
            output_stride *= self.config.stages[i].fft_length

            if i < stage_count - 1:
                vc.memory_barrier()
                vc.barrier()

@lru_cache(maxsize=None)
def make_fft_planner(
        N: int, 
        stride: int = 1,
        batch_y_stride: int = None,
        batch_z_stride: int = None,
        name: str = None,
        max_register_count: int = None) -> FFTPlanner:

    return FFTPlanner(
        N, 
        name=name, 
        batch_y_stride=batch_y_stride,
        batch_z_stride=batch_z_stride,
        fft_stride=stride,
        max_register_count=max_register_count)

def get_planner_cache_info():
    return make_fft_planner.cache_info()

def print_planner_cache_info():
    print(get_cache_info())

def cache_planner_clear():
    return make_fft_planner.cache_clear()