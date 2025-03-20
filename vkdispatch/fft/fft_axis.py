import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from typing import List

from functools import lru_cache

import numpy as np

def int_to_bool_list(x, n):
    """
    Extract the first n bits of integer x (starting from the least significant bit)
    and return them as a list of booleans.
    
    Args:
        x (int): The integer from which to extract bits.
        n (int): The number of bits to extract.
    
    Returns:
        list[bool]: A list of n booleans, where each bool represents a bit of x.
                    The 0th index corresponds to the least significant bit.
    """
    return [(x & (1 << i)) != 0 for i in range(n)]

def bool_list_to_int(bits):
    """
    Convert a list of booleans representing bits to an integer.
    
    Args:
        bits (list[bool]): The list of booleans to convert, where each bool
                           represents a bit. The 0th index is treated as the LSB.
    
    Returns:
        int: The integer obtained from these bits.
    """
    x = 0
    for i, bit in enumerate(bits):
        if bit:
            x |= (1 << i)
    return x

def power_of_2_to_bit_cout(N):
    return int(np.round(np.log2(N)))

def reverse_bits(x, N):
    """
    Reverse the order of the bits in an integer.
    
    Args:
        x (int): The integer to reverse.
        N (int): The number of bits to the power ot 2.
    
    Returns:
        int: The integer obtained by reversing the bits of x.
    """
    n = power_of_2_to_bit_cout(N)

    bits = int_to_bool_list(x, n)
    bits.reverse()
    return bool_list_to_int(bits)

def stockham_shared_buffer(sdata: vc.Buffer[vc.c64], local_vars, output_offset: int, input_offset: int, index: int, N: int, N_total: int):
    vc.memory_barrier()
    vc.barrier()

    for i in range(4):
        # read odd value
        local_vars[i + 4][:] = sdata[input_offset + index*4 + i + N_total // 2]
        # calculate twiddle factor
        local_vars[i][:] = vc.complex_from_euler_angle(-2 * np.pi * (index*4 + i) / N)
        # do multiplication
        local_vars[i + 4][:] = vc.mult_c64(local_vars[i], local_vars[i + 4])
        
        # read even value
        local_vars[i][:] = sdata[input_offset + index*4 + i]
    
    vc.memory_barrier()
    vc.barrier()

    for i in range(4):
        sdata[output_offset + index*4 + i] = local_vars[i] + local_vars[i + 4]
        sdata[output_offset + index*4 + i + N//2] = local_vars[i] - local_vars[i + 4]

class FFTAxisPlanner:
    def __init__(self, N: int, batch_input_stride: int = None, register_count: int = 8, name: str = None):
        if name is None:
            name = f"fft_axis_{N}"
        
        self.N = N
        self.local_size = (max(1, N // register_count), 1, 1)

        if batch_input_stride is None:
            self.batch_input_stride = N

        self.builder = vc.ShaderBuilder(enable_exec_bounds=False)
        old_builder = vc.set_global_builder(self.builder)

        self.signature = vd.ShaderSignature.from_type_annotations(self.builder, [Buff[c64]])
        self.buffer = self.signature.get_variables()[0]

        # Register allocation
        self.register_count = min(self.N, register_count)
        self.registers = [None] * register_count
        for i in range(register_count):
            self.registers[i] = vc.new(c64, var_name=f"register_{i}")

        self.radix_2_even = vc.new(c64, var_name="radix_2_even")
        self.radix_2_odd = vc.new(c64, var_name="radix_2_odd")

        # Local ID within the workgroup
        self.tid = vc.local_invocation().x.copy("tid")

        # Index offset of the current batch
        self.batch_offset = (vc.workgroup().y * self.batch_input_stride).copy("batch_offset")

        self.plan()

        vc.set_global_builder(old_builder)

        self.description = self.builder.build(name)

    def load_buffer_to_registers(self, buffer: Buff[c64], offset: Const[u32], stride: Const[u32], count: int = None, do_bit_reversal: bool = True):
        if count is None:
            count = self.register_count

        for i in range(count):
            register_index = i

            if do_bit_reversal:
                register_index = reverse_bits(i, count)

            self.registers[register_index][:] = buffer[i * stride + offset]

    def store_registers_in_buffer(self, buffer: Buff[c64], offset: Const[u32], stride: Const[u32], count: int = None):
        if count is None:
            count = self.register_count

        for i in range(count):
            register_index = i
            buffer[i * stride + offset] = self.registers[register_index]

    def radix_2_on_registers(self, register_even: int, register_odd: int, index: int, N: int, phase_shift):
        self.radix_2_even[:] = self.registers[register_even]

        self.radix_2_odd.x = -2 * np.pi * (phase_shift + (index / N))
        self.radix_2_odd[:] = vc.complex_from_euler_angle(self.radix_2_odd.x)
        self.radix_2_odd[:] = vc.mult_c64(self.radix_2_odd, self.registers[register_odd])

        self.registers[register_even][:] = self.radix_2_even + self.radix_2_odd
        self.registers[register_odd][:] = self.radix_2_even - self.radix_2_odd
        
        #self.registers[register_even][:] = vc.new_vec2(index, N)
        #self.registers[register_odd][:] = vc.new_vec2(index, N)

    def radix_N_on_registers(self, phase_shift: float = 0, count: int = None, start_stage: int = None):
        if count is None:
            count = self.register_count

        if start_stage is None:
            start_stage = 1

        phase_shift_factor = 1

        for radix_power in range(start_stage, count + 1):
            series_length = 2 ** radix_power
    
            for i in range(count // series_length):
                for j in range(series_length // 2):
                    even_index = i * series_length + j
                    odd_index = i * series_length + j + series_length // 2

                    self.radix_2_on_registers(even_index, odd_index, j, series_length, phase_shift * phase_shift_factor)
            
            phase_shift_factor /= 2
    
    def plan(self):
        sdata = vc.shared_buffer(vc.c64, self.N, "sdata")
        io_offset = vc.new_uint(var_name="io_offset")
        phase_shift = vc.new_float(var_name="phase_shift")
        
        register_bits = power_of_2_to_bit_cout(self.register_count)

        self.load_buffer_to_registers(self.buffer, self.batch_offset + self.tid, self.N // self.register_count)
        self.radix_N_on_registers()
        self.store_registers_in_buffer(sdata, self.tid * self.register_count, 1)

        vc.memory_barrier()
        vc.barrier()
        
        fft_spread = self.register_count

        while fft_spread < self.N:
            radix_count = min(self.register_count, self.N // fft_spread)

            inner_offset = self.tid % fft_spread
            outer_offset = (self.tid / fft_spread) * fft_spread * self.register_count

            io_offset[:] = inner_offset + outer_offset

            phase_shift[:] = inner_offset.cast_to(f32) / (fft_spread * 2)

            start_stage = register_bits - power_of_2_to_bit_cout(radix_count) + 1

            read_stride = fft_spread // (2 ** (start_stage - 1))

            #start_stage = 3

            self.load_buffer_to_registers(sdata, io_offset, read_stride, do_bit_reversal=False)

            vc.memory_barrier()
            vc.barrier()

            self.radix_N_on_registers(phase_shift=phase_shift, start_stage=start_stage)

            if fft_spread * self.register_count < self.N:
                self.store_registers_in_buffer(sdata, io_offset, fft_spread)

                vc.memory_barrier()
                vc.barrier()

            fft_spread *= self.register_count

        self.store_registers_in_buffer(self.buffer, self.batch_offset + self.tid, self.N // self.register_count)
        #self.store_registers_in_buffer(self.buffer, self.batch_offset + self.tid * self.register_count, 1)

@lru_cache(maxsize=None)
def make_fft_stage(
        N: int, 
        stride: int = 1,
        #batch_input_stride: int = None,
        #batch_output_stride: int = None,
        name: str = None):
    
    assert N & (N-1) == 0, "Input length must be a power of 2"
    assert stride == 1, "Only stride 1 is supported for now"

    axis_planner = FFTAxisPlanner(N, name=name)

    return vd.ShaderObject(name, axis_planner.description, axis_planner.signature, local_size=axis_planner.local_size)