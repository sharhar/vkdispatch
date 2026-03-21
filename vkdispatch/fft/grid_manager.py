import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Optional, Tuple, Union, Literal

from .config import FFTConfig
from .prime_utils import prime_factors

from ..compat import numpy_compat as npc

def allocation_valid(workgroup_size: int, shared_memory_size: int):
    valid_workgroup = workgroup_size <= vd.get_context().max_workgroup_invocations
    valid_shared_memory = shared_memory_size <= vd.get_context().max_shared_memory
    return valid_workgroup and valid_shared_memory

def allocate_inline_batches(
        batch_num: int,
        batch_threads: int,
        shared_elements: int,
        element_size: int,
        max_workgroup_size: int,
        max_total_threads: int):
    
    shared_memory_allocation = shared_elements * element_size
    batch_num_primes = prime_factors(batch_num)
    prime_index = 0
    workgroup_size = batch_threads
    inline_batches = 1

    while allocation_valid(workgroup_size, shared_memory_allocation) and \
                            prime_index < len(batch_num_primes) and \
                            inline_batches <= max_workgroup_size and \
                            workgroup_size <= max_total_threads:

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

def allocate_workgroups(total_count: int, declare_variables: bool = True) -> Tuple[vc.ShaderVariable, Tuple[int, int, int]]:
    workgroups_x = set_to_multiple_with_max(
        total_count,
        vd.get_context().max_workgroup_count[0]
    )
    workgroups_y = 1
    workgroups_z = 1

    if not declare_variables:
        return None, (workgroups_x, workgroups_y, workgroups_z)

    workgroup_index = vc.new_uint_register(
        vc.workgroup_id().x,
        var_name="workgroup_index"
    )

    if workgroups_x != total_count:
        workgroups_y = set_to_multiple_with_max(
            total_count // workgroups_x,
            vd.get_context().max_workgroup_count[1]
        )

        workgroup_index += workgroups_x * vc.workgroup_id().y

        if workgroups_y != total_count // workgroups_x:
            workgroups_z = set_to_multiple_with_max(
                total_count // (workgroups_x * workgroups_y),
                vd.get_context().max_workgroup_count[2]
            )

            workgroup_index += workgroups_x * workgroups_y * vc.workgroup_id().z

    return workgroup_index, (workgroups_x, workgroups_y, workgroups_z)

def decompose_workgroup_index(
        workgroup_index: vc.ShaderVariable,
        inner_batch_count: int,
        fft_threads: int,
        local_size: Tuple[int, int, int]) -> Tuple[vc.ShaderVariable, vc.ShaderVariable]:

    if inner_batch_count == None:
        if fft_threads == 1:
            return None, workgroup_index * local_size[0] + vc.local_invocation_id().x

        return None, workgroup_index * local_size[1] + vc.local_invocation_id().y 

    global_inner_offset = vc.new_uint_register(
        (workgroup_index % inner_batch_count) * local_size[0] + vc.local_invocation_id().x,
        var_name="global_inner_index"
    )

    global_outer_offset = vc.new_uint_register(
        (workgroup_index // inner_batch_count) * local_size[2] + vc.local_invocation_id().z,
        var_name="global_outer_index"
    )

    return global_inner_offset, global_outer_offset

class FFTGridManager:
    config: FFTConfig

    shared_memory_enabled: bool
    shared_memory_allocation: int

    inline_batches_inner: int
    inline_batches_outer: int

    local_inner: Optional[vc.ShaderVariable]
    local_outer: vc.ShaderVariable

    tid: vc.ShaderVariable

    global_inner_offset: Union[vc.ShaderVariable, Literal[0]]
    global_outer_offset: vc.ShaderVariable

    local_size: Tuple[int, int, int]
    workgroup_count: Tuple[int, int, int]
    exec_size: Tuple[int, int, int]

    workgroup_index: vc.ShaderVariable

    transposed_offset: Optional[vc.ShaderVariable]
    transposed_stride: int

    transposed_inner_offset: Optional[vc.ShaderVariable]
    transposed_inner_stride: int

    def __init__(self, config: FFTConfig, force_sdata: bool = False, declare_variables: bool = True):
        self.config = config

        make_sdata_buffer = config.batch_threads > 1 or force_sdata

        self.inline_batches_inner = allocate_inline_batches(
            config.batch_inner_count,
            config.batch_threads,
            config.sdata_allocation if make_sdata_buffer else 0,
            config.compute_type.item_size,
            min(vd.get_context().max_workgroup_size[0], 4),
            vd.get_context().max_workgroup_invocations)
        
        max_inline_outer_batches = vd.get_context().max_workgroup_size[
            1 if config.batch_inner_count == 1 else 2
        ]

        # For some reason it's better not to have too many inline outer batches
        max_inline_outer_batches = min(max_inline_outer_batches, vd.get_context().subgroup_size)

        self.inline_batches_outer = allocate_inline_batches(
            config.batch_outer_count,
            config.batch_threads * self.inline_batches_inner,
            config.sdata_allocation * self.inline_batches_inner if make_sdata_buffer else 0,
            config.compute_type.item_size,
            vd.get_context().max_workgroup_size[
                1 if self.inline_batches_inner == 1 else 2
            ],
            max_inline_outer_batches)


        if config.batch_inner_count > 1:
            self.local_size = (self.inline_batches_inner, config.batch_threads, self.inline_batches_outer)

            inner_workgroups = config.batch_inner_count // self.inline_batches_inner
            outer_workgroups = config.batch_outer_count // self.inline_batches_outer
            
            self.workgroup_index, self.workgroup_count = allocate_workgroups(
                inner_workgroups * outer_workgroups,
                declare_variables=declare_variables
            )

            if declare_variables:
                self.local_inner = vc.local_invocation_id().x
                self.local_outer = vc.local_invocation_id().z

                self.global_inner_offset, self.global_outer_offset = decompose_workgroup_index(
                    self.workgroup_index,
                    inner_workgroups,
                    config.batch_threads,
                    self.local_size
                )
                
                self.tid = vc.local_invocation_id().y.to_register("tid")
        else:
            self.local_inner = None
            self.global_inner_offset = 0

            if config.batch_threads > 1:
                self.local_size = (config.batch_threads, self.inline_batches_outer, 1)
            else:
                self.local_size = (self.inline_batches_outer, 1, 1)

            self.workgroup_index, self.workgroup_count = allocate_workgroups(
                config.batch_outer_count // self.inline_batches_outer,
                declare_variables=declare_variables
            )
            
            if declare_variables:
                if config.batch_threads > 1:
                    self.tid = vc.local_invocation_id().x.to_register("tid")
                    self.local_outer = vc.local_invocation_id().y
                else:
                    self.tid = 0
                    self.local_outer = vc.local_invocation_id().x

                _, self.global_outer_offset = decompose_workgroup_index(
                    self.workgroup_index,
                    None,
                    config.batch_threads,
                    self.local_size
                )

        self.exec_size = (
            self.local_size[0] * self.workgroup_count[0],
            self.local_size[1] * self.workgroup_count[1],
            self.local_size[2] * self.workgroup_count[2]
        )

        if not declare_variables:
            return

        self.transposed_stride = npc.prod(self.local_size)
        self.transposed_offset = vc.local_invocation_index() + self.transposed_stride * self.config.register_count * self.workgroup_index
        
        self.transposed_inner_stride = None
        self.transposed_inner_offset = None

        if config.batch_inner_count > 1:
            self.transposed_inner_stride = self.local_size[0] * self.local_size[1]
            self.transposed_inner_offset = vc.local_invocation_id().x + self.local_size[0] * vc.local_invocation_id().y + \
                                            self.transposed_inner_stride * self.config.register_count * (self.workgroup_index % inner_workgroups)
        else:
            self.transposed_inner_stride = self.local_size[0]
            self.transposed_inner_offset = vc.local_invocation_id().x

    def get_transposed_index(self, register_id: int, inner_only: bool = False) -> vc.ShaderVariable:
        if not inner_only:
            return self.transposed_offset + register_id * self.transposed_stride

        return self.transposed_inner_offset + register_id * self.transposed_inner_stride
