import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from typing import List, Tuple

from functools import lru_cache

import numpy as np

from .prime_utils import prime_factors, group_primes

class FFTAxisPlanner:
    def __init__(self, N: int, batch_y_stride: int = None, batch_z_stride: int = None, fft_stride: int = None, max_register_count: int = None, name: str = None):
        if name is None:
            name = f"fft_axis_{N}"
        
        self.N = N
        self.prime_groups = group_primes(prime_factors(N), max_register_count if max_register_count is not None else 16)
        self.group_sizes = [int(np.prod(group)) for group in self.prime_groups]
        self.register_count = max(self.group_sizes)

        self.local_size = (N // min(self.group_sizes), 1, 1)
        
        self.fft_stride = fft_stride if fft_stride is not None else 1

        self.builder = vc.ShaderBuilder(enable_exec_bounds=False)
        old_builder = vc.set_global_builder(self.builder)

        self.signature = vd.ShaderSignature.from_type_annotations(self.builder, [Buff[c64]])
        self.buffer = self.signature.get_variables()[0]

        self.batch_y_stride = batch_y_stride if batch_y_stride is not None else N
        self.batch_z_stride = batch_z_stride if batch_z_stride is not None else 1
        
        # Register allocation
        self.registers = [vc.new(c64, 0, var_name=f"register_{i}") for i in range(self.register_count)]
        self.radix_registers = [vc.new(c64, 0, var_name=f"radix_{i}") for i in range(self.register_count)]
        self.omega_register = vc.new(c64, 0, var_name="omega_register")

        # Local ID within the workgroup
        self.tid = vc.local_invocation().x.copy("tid")

        # Index offset of the current batch
        self.batch_offset = (vc.workgroup().y * self.batch_y_stride + vc.workgroup().z * self.batch_z_stride).copy("batch_offset")

        self.sdata = vc.shared_buffer(vc.c64, self.N, "sdata")

        self.plan()

        vc.set_global_builder(old_builder)

        self.description = self.builder.build(name)

    def load_buffer_to_registers(self, buffer: Buff[c64], offset: Const[u32], stride: Const[u32], count: int = None):
        if count is None:
            count = self.register_count

        batch_offset = self.batch_offset
        fft_stride = self.fft_stride

        if buffer is None:
            buffer = self.sdata
            batch_offset = 0
            fft_stride = 1

        vc.comment(f"Loading {count} registers from buffer at offset {offset} and stride {stride}")

        for i in range(count):
            self.registers[i][:] = buffer[(i * stride + offset) * fft_stride  + batch_offset]

    def store_registers_in_buffer(self, buffer: Buff[c64], offset: Const[u32], stride: Const[u32], count: int = None):
        if count is None:
            count = self.register_count

        batch_offset = self.batch_offset
        fft_stride = self.fft_stride

        if buffer is None:
            buffer = self.sdata
            batch_offset = 0
            fft_stride = 1

        vc.comment(f"Storing {count} registers to buffer at offset {offset} and stride {stride}")

        for i in range(count):
            buffer[(i * stride + offset) * fft_stride + batch_offset] = self.registers[i]
    
    def radix_P(self, register_list: List[vc.ShaderVariable]):
        assert len(register_list) <= len(self.radix_registers), "Too many registers for radix_P"

        if len(register_list) == 1:
            return

        vc.comment(f"Performing a DFT for Radix-{len(register_list)} FFT")

        for i in range(0, len(register_list)):
            self.radix_registers[i].x = 0
            self.radix_registers[i].y = 0

            for j in range(0, len(register_list)):
                if i == 0 or j == 0:
                    self.radix_registers[i] += register_list[j]
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
            if isinstance(twiddle_index, int):
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

    def process_prime_group(self, primes: List[int], output_stride: int, input = None, output = None):
        group_size = np.prod(primes)

        vc.comment(f"Processing prime group {primes} by doing radix-{group_size} FFT on {self.N // group_size} groups")
        vc.if_statement(self.tid < self.N // group_size)

        block_width = output_stride * group_size

        inner_block_offset = self.tid % output_stride
        block_index = (self.tid * group_size) / block_width
        sub_sequence_offset = block_index * block_width + inner_block_offset
        
        self.load_buffer_to_registers(input, self.tid, self.N // group_size, count=group_size)
        
        self.apply_cooley_tukey_twiddle_factors(self.registers[:group_size], twiddle_index=inner_block_offset, twiddle_N=block_width)
        self.registers[:group_size] = self.register_radix_composite(self.registers[:group_size], primes)

        vc.end()

        if input is None and output is None:
            vc.memory_barrier()
            vc.barrier()

        vc.if_statement(self.tid < self.N // group_size)

        self.store_registers_in_buffer(output, sub_sequence_offset, output_stride, count=group_size)

        vc.end()

    def plan(self):
        output_stride = 1

        for i in range(len(self.prime_groups)):
            self.process_prime_group(
                self.prime_groups[i],
                output_stride,
                input=self.buffer if i == 0 else None,
                output=self.buffer if i == len(self.prime_groups) - 1 else None)
            
            output_stride *= self.group_sizes[i]

            if i < len(self.prime_groups) - 1:
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

    return vd.ShaderObject(axis_planner.description, axis_planner.signature, local_size=axis_planner.local_size)

def get_cache_info():
    return make_fft_stage.cache_info()

def print_cache_info():
    print(get_cache_info())

def cache_clear():
    return make_fft_stage.cache_clear()