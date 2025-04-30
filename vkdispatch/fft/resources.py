import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np
import dataclasses
from typing import List, Tuple

from .config import FFTConfig
from .prime_utils import prime_factors, DEFAULT_REGISTER_LIMIT

def allocation_valid(workgroup_size: int, shared_memory: int):
    return workgroup_size <= vd.get_context().max_workgroup_invocations and shared_memory <= vd.get_context().max_shared_memory

def allocate_inline_batches(batch_num: int, batch_threads: int, N: int, max_workgroup_size: int, max_total_threads: int):
    batch_num_primes = prime_factors(batch_num)

    prime_index = len(batch_num_primes) - 1

    workgroup_size = batch_threads
    shared_memory_allocation = N * vd.complex64.item_size
    inline_batches = 1

    while allocation_valid(workgroup_size, shared_memory_allocation) and prime_index >= 0 and inline_batches <= max_workgroup_size and workgroup_size <= max_total_threads:
        test_prime = batch_num_primes[prime_index]

        if allocation_valid(workgroup_size * test_prime, shared_memory_allocation * test_prime) and inline_batches * test_prime <= max_workgroup_size and workgroup_size * test_prime <= max_total_threads:
            workgroup_size *= test_prime
            shared_memory_allocation *= test_prime
            inline_batches *= test_prime
        
        prime_index -= 1

    return inline_batches

@dataclasses.dataclass
class FFTResources:
    registers: List[vc.ShaderVariable]
    radix_registers: List[vc.ShaderVariable]
    omega_register: vc.ShaderVariable
    tid: Const[u32]
    input_batch_offset: Const[u32]
    output_batch_offset: Const[u32]
    subsequence_offset: Const[u32]
    sdata: Buff[c64]
    sdata_offset: Const[u32]
    io_index: Const[u32]
    io_index_2: Const[u32]

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

def allocate_fft_resources(config: FFTConfig) -> FFTResources:
    inline_batch_z = allocate_inline_batches(config.batch_z_count, config.batch_threads, config.sdata_allocation, vd.get_context().max_workgroup_size[0], vd.get_context().max_workgroup_invocations)
    inline_batch_y = allocate_inline_batches(config.batch_y_count, config.batch_threads * inline_batch_z, config.sdata_allocation * inline_batch_z, vd.get_context().max_workgroup_size[2], vd.get_context().subgroup_size)

    sdata_buffer = vc.shared_buffer(vc.c64, config.sdata_allocation * inline_batch_y * inline_batch_z, "sdata")
    sdata_offset = None

    if inline_batch_y > 1 or inline_batch_z > 1:
        sdata_offset = vc.new_uint(
            vc.local_invocation().z * inline_batch_z * config.N + vc.local_invocation().x * config.N,
            var_name="sdata_offset")

    resources = FFTResources(
        registers=[vc.new(c64, 0, var_name=f"register_{i}") for i in range(config.register_count)],
        radix_registers=[vc.new(c64, 0, var_name=f"radix_{i}") for i in range(config.max_prime_radix)],
        omega_register=vc.new(c64, 0, var_name="omega_register"),
        tid=vc.local_invocation().y.copy("tid"),
        input_batch_offset=vc.new_uint(var_name="input_batch_offset"),
        output_batch_offset=vc.new_uint(var_name="output_batch_offset"),
        subsequence_offset=vc.new_uint(0, var_name="subsequence_offset"),
        sdata=sdata_buffer,
        sdata_offset=sdata_offset,
        io_index=vc.new_uint(0, var_name="io_index"),
        io_index_2=vc.new_uint(0, var_name="io_index_2"),
        inline_batch_y=inline_batch_y,
        inline_batch_z=inline_batch_z,
        shared_memory_size=config.N * inline_batch_y * inline_batch_z * vd.complex64.item_size,
        local_size=(inline_batch_z, config.batch_threads, inline_batch_y)
    )

    #resources.reset()

    return resources

