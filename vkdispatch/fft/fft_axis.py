import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

from typing import List, Tuple

from functools import lru_cache

import numpy as np

def prime_factors(n):
    factors = []
    
    # Handle the factor 2 separately
    while n % 2 == 0:
        factors.append(2)
        n //= 2
        
    # Now handle odd factors
    factor = 3
    while factor * factor <= n:
        while n % factor == 0:
            factors.append(factor)
            n //= factor
        factor += 2
        
    # If at the end, n is greater than 1, it is a prime number itself
    if n > 1:
        factors.append(n)
        
    return factors

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
    def __init__(self, N: int, batch_input_stride: int = None, register_count: int = 16, name: str = None):
        if name is None:
            name = f"fft_axis_{N}"
        
        self.N = N
        #self.local_size = (max(1, N // register_count), 1, 1)
        self.local_size = (1, 1, 1)

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

        self.omega_register = vc.new(c64, var_name="omega_register")

        self.radix_registers = [None] * register_count
        for i in range(16):
            self.radix_registers[i] = vc.new(c64, var_name=f"radix_{i}")

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
    
    def radix_P(self, register_list: List[vc.ShaderVariable]):
        assert len(register_list) <= len(self.radix_registers), "Too many registers for radix_P"

        if len(register_list) == 1:
            return
        
        for i in range(0, len(register_list)):
            self.radix_registers[i].x = 0
            self.radix_registers[i].y = 0

            for j in range(0, len(register_list)):
                if i == 0 or j == 0:
                    self.omega_register[:] = register_list[j]
                else:
                    self.omega_register.x = -2 * np.pi * i * j / len(register_list)
                    self.omega_register[:] = vc.complex_from_euler_angle(self.omega_register.x)
                    self.omega_register[:] = vc.mult_c64(self.omega_register, register_list[j])

                self.radix_registers[i][:] = self.radix_registers[i] + self.omega_register

        for i in range(0, len(register_list)):
            register_list[i][:] = self.radix_registers[i]

    def apply_cooley_tukey_twiddle_facotrs(self, register_list: List[vc.ShaderVariable], twiddle_index: int = 0, twiddle_N: int = 1):
        for i in range(len(register_list)):
            self.omega_register.x = -2 * np.pi * i * twiddle_index / twiddle_N

            self.omega_register[:] = vc.complex_from_euler_angle(self.omega_register.x)
            self.omega_register[:] = vc.mult_c64(self.omega_register, register_list[i])

            register_list[i][:] = self.omega_register

    def radix_composite(self, register_list: List[vc.ShaderVariable], primes: List[int]):
        if len(register_list) == 1:
            return
        
        N = len(register_list)

        assert N == np.prod(primes), "Product of primes must be equal to the number of registers"

        output_stride = 1

        for prime in primes:
            sub_squences = [register_list[i::N//prime] for i in range(N//prime)]

            block_width = output_stride * prime

            for i in range(0, N // prime):
                inner_block_offset = i % output_stride
                block_index = i * prime // block_width

                self.apply_cooley_tukey_twiddle_facotrs(sub_squences[i], twiddle_index=inner_block_offset, twiddle_N=block_width)
                self.radix_P(sub_squences[i])
                
                sub_sequence_offset = block_index * block_width + inner_block_offset

                for j in range(prime):
                    register_list[sub_sequence_offset + j * output_stride] = sub_squences[i][j]
            
            output_stride *= prime   

        return register_list

    def plan(self):
        sdata = vc.shared_buffer(vc.c64, self.N, "sdata")

        self.load_buffer_to_registers(self.buffer, self.batch_offset + self.tid, self.N // self.register_count, do_bit_reversal=False)

        self.registers[:self.N] = self.radix_composite(self.registers[:self.N], prime_factors(self.N))
        
        #self.radix_P(self.registers[:self.N])
        
        #self.store_registers_in_buffer(sdata, self.tid * self.register_count, 1)

        #vc.memory_barrier()
        #vc.barrier()

        #self.store_registers_in_buffer(self.buffer, self.batch_offset + self.tid, self.N // self.register_count)

        #self.load_buffer_to_registers(sdata, self.tid, self.N // self.register_count, do_bit_reversal=False)
        #self.apply_cooley_tukey_twiddle_facotrs(self.registers, twiddle_index=self.tid, twiddle_N=64)
        #self.radix_composite(self.registers, [2, 2, 2])

        #self.apply_twiddle_facotrs(self.registers[:4], twiddle_index=self.tid, twiddle_N=32)
        #self.registers[:4] = self.radix_composite(self.registers[:4], [2, 2])

        #self.apply_twiddle_facotrs(self.registers[4:], twiddle_index=self.tid, twiddle_N=32)
        #self.registers[4:] = self.radix_composite(self.registers[4:], [2, 2])
        
        #self.store_registers_in_buffer(sdata, self.tid, self.register_count)

        #vc.memory_barrier()
        #vc.barrier()

        #self.load_buffer_to_registers(sdata, self.tid, self.N // self.register_count, do_bit_reversal=False)
        #self.apply_twiddle_facotrs(self.registers, twiddle_index=self.tid, twiddle_N=64)
        #self.radix_composite(self.registers, [2, 2, 2])

        #self.store_registers_in_buffer(self.buffer, self.batch_offset + self.tid, self.register_count)

        self.store_registers_in_buffer(self.buffer, self.batch_offset + self.tid * self.register_count, 1)
        
        #

@lru_cache(maxsize=None)
def make_fft_stage(
        N: int, 
        stride: int = 1,
        #batch_input_stride: int = None,
        #batch_output_stride: int = None,
        name: str = None):
    
    #assert N & (N-1) == 0, "Input length must be a power of 2"
    assert stride == 1, "Only stride 1 is supported for now"

    axis_planner = FFTAxisPlanner(N, name=name)

    return vd.ShaderObject(name, axis_planner.description, axis_planner.signature, local_size=axis_planner.local_size)