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
    global_inner_index: Const[u32]
    global_outer_index: Const[u32]

    shared_memory_size: int
    local_size: Tuple[int, int, int]

    def reset(self):
        for register in self.registers:
            register[:] = "vec2(0)"
        
        for register in self.radix_registers:
            register[:] = "vec2(0)"
        
        self.omega_register[:] = "vec2(0)"

def allocate_fft_resources(config: FFTConfig) -> FFTResources:
    inline_batch_inner = allocate_inline_batches(
        config.batch_inner_count,
        config.batch_threads,
        config.sdata_allocation,
        vd.get_context().max_workgroup_size[0],
        vd.get_context().max_workgroup_invocations)
    
    inline_batch_outer = allocate_inline_batches(
        config.batch_outer_count,
        config.batch_threads * inline_batch_inner,
        config.sdata_allocation * inline_batch_inner,
        vd.get_context().max_workgroup_size[1 if inline_batch_inner == 1 else 2],
        vd.get_context().subgroup_size)

    sdata_buffer = vc.shared_buffer(vc.c64, config.sdata_allocation * inline_batch_outer * inline_batch_inner, "sdata")
    sdata_offset = None

    local_inner = None
    global_inner = 0

    local_outer = vc.local_invocation().y
    global_outer = vc.global_invocation().y

    tid = vc.local_invocation().x
    local_size = (config.batch_threads, inline_batch_outer, 1)

    if config.batch_inner_count > 1:
        local_inner = vc.local_invocation().x
        global_inner = vc.global_invocation().x

        local_outer = vc.local_invocation().z
        global_outer = vc.global_invocation().z

        local_size = (inline_batch_inner, config.batch_threads, inline_batch_outer)
        tid = vc.local_invocation().y

    if inline_batch_outer > 1 or inline_batch_inner > 1:
        sdata_offset_value = local_outer * inline_batch_inner * config.N

        if local_inner is not None:
            sdata_offset_value = sdata_offset_value + local_inner * config.N

        sdata_offset = vc.new_uint(sdata_offset_value, var_name="sdata_offset")

    resources = FFTResources(
        registers=[vc.new(c64, 0, var_name=f"register_{i}") for i in range(config.register_count)],
        radix_registers=[vc.new(c64, 0, var_name=f"radix_{i}") for i in range(config.max_prime_radix)],
        omega_register=vc.new(c64, 0, var_name="omega_register"),
        tid=tid.copy("tid"),
        input_batch_offset=vc.new_uint(var_name="input_batch_offset"),
        output_batch_offset=vc.new_uint(var_name="output_batch_offset"),
        subsequence_offset=vc.new_uint(0, var_name="subsequence_offset"),
        sdata=sdata_buffer,
        sdata_offset=sdata_offset,
        io_index=vc.new_uint(0, var_name="io_index"),
        io_index_2=vc.new_uint(0, var_name="io_index_2"),
        shared_memory_size=config.N * inline_batch_outer * inline_batch_inner * vd.complex64.item_size,
        local_size=local_size, #(inline_batch_inner, config.batch_threads, inline_batch_outer),
        global_inner_index=global_inner,
        global_outer_index=global_outer
    )



    #resources.reset()

    return resources

