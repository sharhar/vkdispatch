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

def allocate_fft_resources(config: FFTConfig) -> FFTResources:
    inline_batch_z = allocate_inline_batches(config.batch_z_count, config.batch_threads, config.N, vd.get_context().max_workgroup_size[2])
    inline_batch_y = allocate_inline_batches(config.batch_y_count, config.batch_threads * inline_batch_z, config.N, vd.get_context().max_workgroup_size[1])

    resources = FFTResources(
        registers=[vc.new(c64, 0, var_name=f"register_{i}") for i in range(config.register_count)],
        radix_registers=[vc.new(c64, 0, var_name=f"radix_{i}") for i in range(config.register_count)],
        omega_register=vc.new(c64, 0, var_name="omega_register"),
        tid=vc.local_invocation().x.copy("tid"),
        input_batch_offset=vc.new_uint(var_name="input_batch_offset"),
        output_batch_offset=vc.new_uint(var_name="output_batch_offset"),
        subsequence_offset=vc.new_uint(0, var_name="subsequence_offset"),
        sdata=vc.shared_buffer(vc.c64, config.N * inline_batch_y * inline_batch_z, "sdata"),
        sdata_offset=(vc.local_invocation().y * inline_batch_z * config.N + vc.local_invocation().z * config.N).copy("sdata_offset"),
        io_index=vc.new_int(0, var_name="io_index"),
        inline_batch_y=inline_batch_y,
        inline_batch_z=inline_batch_z,
        shared_memory_size=config.N * inline_batch_y * inline_batch_z * vd.complex64.item_size,
        local_size=(config.batch_threads, inline_batch_y, inline_batch_z)
    )

    resources.reset()

    return resources

