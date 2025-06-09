import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np
import dataclasses
from typing import List, Tuple

from .config import FFTConfig
from .prime_utils import prime_factors, default_register_limit

def allocation_valid(workgroup_size: int, shared_memory: int):
    return workgroup_size <= vd.get_context().max_workgroup_invocations and shared_memory <= vd.get_context().max_shared_memory

def allocate_inline_batches(batch_num: int, batch_threads: int, N: int, max_workgroup_size: int, max_total_threads: int):
    shared_memory_allocation = N * vd.complex64.item_size
    batch_num_primes = prime_factors(batch_num)
    prime_index = 0
    workgroup_size = batch_threads
    inline_batches = 1

    while allocation_valid(workgroup_size, shared_memory_allocation) and prime_index < len(batch_num_primes) and inline_batches <= max_workgroup_size and workgroup_size <= max_total_threads:
        test_prime = batch_num_primes[prime_index]

        is_valid = allocation_valid(workgroup_size * test_prime, shared_memory_allocation * test_prime)

        is_valid = is_valid and inline_batches * test_prime <= max_workgroup_size
        is_valid = is_valid and workgroup_size * test_prime <= max_total_threads

        if is_valid:
            workgroup_size *= test_prime
            shared_memory_allocation *= test_prime
            inline_batches *= test_prime
        
        prime_index += 1

    return inline_batches


def allocate_workgroups(total_count: int) -> Tuple[vc.ShaderVariable, Tuple[int, int, int]]:
    def set_to_multiple_with_max(count, max_count):
        if count <= max_count:
            return count
        
        count_primes = prime_factors(count)

        result_count = 1
        for prime in count_primes:
            if result_count * prime > max_count:
                break
            result_count *= prime

        return result_count
    
    workgroups_x = set_to_multiple_with_max(
        total_count,
        vd.get_context().max_workgroup_count[0]
    )
    workgroups_y = 1
    workgroups_z = 1

    workgroup_index = vc.new_uint(
        vc.workgroup().x,
        var_name="workgroup_index"
    )

    if workgroups_x != total_count:
        workgroups_y = set_to_multiple_with_max(
            total_count // workgroups_x,
            vd.get_context().max_workgroup_count[1]
        )

        workgroup_index += workgroups_x * vc.workgroup().y

        if workgroups_y != total_count // workgroups_x:
            workgroups_z = set_to_multiple_with_max(
                total_count // (workgroups_x * workgroups_y),
                vd.get_context().max_workgroup_count[2]
            )

            workgroup_index += workgroups_x * workgroups_y * vc.workgroup().z

    return workgroup_index, (workgroups_x, workgroups_y, workgroups_z)

def decompose_workgroup_index(workgroup_index: vc.ShaderVariable, inner_batch_count: int, fft_threads: int, local_size: Tuple[int, int, int]) -> Tuple[vc.ShaderVariable, vc.ShaderVariable]:
    if inner_batch_count == None:
        if fft_threads == 1:
            return None, workgroup_index * local_size[0] + vc.local_invocation().x

        return None, workgroup_index * local_size[1] + vc.local_invocation().y 

    global_inner = vc.new_uint(
        (workgroup_index % inner_batch_count) * local_size[0] + vc.local_invocation().x,
        var_name="global_inner_index"
    )

    global_outer = vc.new_uint(
        (workgroup_index / inner_batch_count) * local_size[2] + vc.local_invocation().z,
        var_name="global_outer_index"
    )

    return global_inner, global_outer

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
    exec_size: Tuple[int, int, int]

    shared_memory_size: int
    local_size: Tuple[int, int, int]

def allocate_fft_resources(config: FFTConfig, convolve: bool) -> FFTResources:
    make_sdata_buffer = config.batch_threads > 1 or convolve

    inline_batch_inner = allocate_inline_batches(
        config.batch_inner_count,
        config.batch_threads,
        config.sdata_allocation if make_sdata_buffer else 0,
        vd.get_context().max_workgroup_size[0],
        vd.get_context().max_workgroup_invocations)

    max_inline_outer_batches = vd.get_context().max_workgroup_size[1 if config.batch_inner_count == 1 else 2]

    # For some reason it's better not to have too many inline outer batches
    max_inline_outer_batches = min(max_inline_outer_batches, vd.get_context().subgroup_size)

    inline_batch_outer = allocate_inline_batches(
        config.batch_outer_count,
        config.batch_threads * inline_batch_inner,
        config.sdata_allocation * inline_batch_inner if make_sdata_buffer else 0,
        vd.get_context().max_workgroup_size[1 if inline_batch_inner == 1 else 2],
        max_inline_outer_batches)

    sdata_buffer = None

    if make_sdata_buffer:
        sdata_buffer = vc.shared_buffer(
            vd.complex64,
            config.sdata_allocation * inline_batch_outer * inline_batch_inner,
            var_name="sdata")


    if config.batch_inner_count > 1:
        local_inner = vc.local_invocation().x
        local_outer = vc.local_invocation().z
        local_size = (inline_batch_inner, config.batch_threads, inline_batch_outer)

        inner_workgroups = config.batch_inner_count // inline_batch_inner
        outer_workgroups = config.batch_outer_count // inline_batch_outer
        
        workgroup_index, workgroups = allocate_workgroups(inner_workgroups * outer_workgroups)

        global_inner, global_outer = decompose_workgroup_index(
            workgroup_index,
            inner_workgroups,
            config.batch_threads,
            local_size
        )

        exec_size = (
            local_size[0] * workgroups[0],
            local_size[1] * workgroups[1],
            local_size[2] * workgroups[2]
        )
        
        tid = vc.local_invocation().y.copy("tid")
    else:
        local_inner = None
        global_inner = 0

        if config.batch_threads > 1:
            tid = vc.local_invocation().x.copy("tid")
            local_outer = vc.local_invocation().y
            local_size = (config.batch_threads, inline_batch_outer, 1)
        else:
            tid = vc.new_uint(0, var_name="tid")
            local_outer = vc.local_invocation().x
            local_size = (inline_batch_outer, 1, 1)

        workgroup_index, workgroups = allocate_workgroups(config.batch_outer_count // inline_batch_outer)

        _, global_outer = decompose_workgroup_index(workgroup_index, None, config.batch_threads, local_size)

        exec_size = (
            local_size[0] * workgroups[0],
            local_size[1] * workgroups[1],
            local_size[2] * workgroups[2]
        )

    sdata_offset = None
    
    if inline_batch_outer > 1 or inline_batch_inner > 1:
        sdata_offset_value = local_outer * inline_batch_inner * config.N

        if local_inner is not None:
            sdata_offset_value = sdata_offset_value + local_inner * config.N

        sdata_offset = vc.new_uint(sdata_offset_value, var_name="sdata_offset")

    resources = FFTResources(
        registers=[vc.new(c64, 0, var_name=f"register_{i}") for i in range(config.register_count)],
        radix_registers=[vc.new(c64, 0, var_name=f"radix_{i}") for i in range(config.max_prime_radix)],
        omega_register=vc.new(c64, 0, var_name="omega_register"),
        tid=tid,
        input_batch_offset=vc.new_uint(var_name="input_batch_offset"),
        output_batch_offset=vc.new_uint(var_name="output_batch_offset"),
        subsequence_offset=vc.new_uint(0, var_name="subsequence_offset"),
        sdata=sdata_buffer,
        sdata_offset=sdata_offset,
        io_index=vc.new_uint(0, var_name="io_index"),
        io_index_2=vc.new_uint(0, var_name="io_index_2"),
        shared_memory_size=config.N * inline_batch_outer * inline_batch_inner * vd.complex64.item_size,
        local_size=local_size,
        global_inner_index=global_inner,
        global_outer_index=global_outer,
        exec_size=exec_size
    )

    return resources

