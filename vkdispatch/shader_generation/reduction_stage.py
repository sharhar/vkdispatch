import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Callable
from typing import List

import dataclasses

@dataclasses.dataclass
class ReductionParams:
    input_offset: vd.int32
    input_size: vd.int32
    input_stride: vd.int32
    input_y_batch_stride: vd.int32
    input_z_batch_stride: vd.int32

    output_offset: vd.int32
    output_stride: vd.int32
    output_y_batch_stride: vd.int32
    output_z_batch_stride: vd.int32

def global_reduce(
        reduction: vd.ReductionOperation, 
        out_type: vd.dtype, 
        buffers: List[vc.BufferVariable], 
        params: ReductionParams,
        map_func: Callable = None):
    
    ind = (vc.global_invocation().x * params.input_stride).copy("ind")
    reduction_aggregate = vc.new(out_type, reduction.identity, var_name="reduction_aggregate")

    batch_offset = vc.workgroup().y * params.input_y_batch_stride
    inside_batch_offset = vc.workgroup().z * params.input_z_batch_stride

    start_index = vc.new_uint(params.input_offset + inside_batch_offset + batch_offset, var_name="start_index")

    current_index = vc.new_uint(start_index + ind, var_name="current_index")

    end_index = vc.new_uint(start_index + params.input_size, var_name="end_index")

    vc.while_statement(current_index < end_index)

    mapped_value = buffers[0][current_index]


    if map_func is not None:
        vc.set_mapping_index(current_index)
        mapped_value = map_func(*buffers)

    reduction_aggregate[:] = reduction.reduction(reduction_aggregate, mapped_value)

    current_index += vc.workgroup_size().x * vc.num_workgroups().x

    vc.end()

    return reduction_aggregate

def workgroup_reduce(
        reduction_aggregate: vc.ShaderVariable,
        reduction: vd.ReductionOperation,
        out_type: vd.dtype,
        group_size: int):
    tid = vc.local_invocation().x
    
    sdata = vc.shared_buffer(out_type, group_size, var_name="sdata")

    sdata[tid] = reduction_aggregate

    vc.memory_barrier()
    vc.barrier()
    
    current_size = group_size // 2
    while current_size > vd.get_context().subgroup_size:
        vc.if_statement(tid < current_size)
        sdata[tid] = reduction.reduction(sdata[tid], sdata[tid + current_size])            
        if current_size // 2 > vd.get_context().subgroup_size:
            vc.end()
        else:
            vc.else_if_statement(tid < 2*vc.subgroup_size())
            sdata[tid] = vc.new(out_type, 0)
            vc.end()
        
        vc.memory_barrier()
        vc.barrier()
        
        current_size //= 2

    return sdata

def subgroup_reduce(
        sdata: vc.ShaderVariable,
        reduction: vd.ReductionOperation,
        group_size: int):
    tid = vc.local_invocation().x
    subgroup_size = vd.get_context().subgroup_size

    if group_size > subgroup_size:
        vc.if_all(tid < subgroup_size)
        sdata[tid] = reduction.reduction(sdata[tid], sdata[tid + subgroup_size])
        vc.end()
        vc.subgroup_barrier()
    
    
    if reduction.subgroup_reduction is not None:
        local_var = sdata[tid].copy("local_var")
        local_var[:] = reduction.subgroup_reduction(local_var)

        return local_var
    else:
        current_size = subgroup_size // 2
        while current_size > 1:
            vc.if_statement(tid < current_size)
            sdata[tid] = reduction.reduction(sdata[tid], sdata[tid + current_size])
            vc.end()
            vc.subgroup_barrier()
            
            current_size //= 2
        
        result = reduction.reduction(sdata[tid], sdata[tid + current_size])

        return result

def make_reduction_stage(
        reduction: vd.ReductionOperation, 
        out_type: vd.dtype, 
        group_size: int, 
        output_is_input: bool,
        name: str = None,
        map_func: Callable = None,
        input_types: List = None) -> vd.ShaderObject:

    if name is None:
        name = f"reduction_stage_{reduction.name}_{out_type.name}_{input_types}_{group_size}"

    builder = vc.ShaderBuilder()
    old_builder = vc.set_global_builder(builder)
    
    signature_type_array = []
    
    signature_type_array.append(vc.Buffer[out_type])

    if input_types is not None:
        signature_type_array.extend(input_types)

    signature_type_array.append(ReductionParams)

    signature = vd.ShaderSignature.from_type_annotations(builder, signature_type_array)
    input_variables = signature.get_variables()

    params: ReductionParams = input_variables[-1]

    input_buffers = input_variables[:-1] if output_is_input else input_variables[1:-1]

    reduction_aggregate = global_reduce(reduction, out_type, input_buffers, params, map_func)
    sdata = workgroup_reduce(reduction_aggregate, reduction, out_type, group_size)
    local_var = subgroup_reduce(sdata, reduction, group_size)

    batch_offset = vc.workgroup().y * params.output_y_batch_stride
    output_offset = vc.workgroup().x * params.output_stride

    vc.if_statement(vc.local_invocation().x == 0)
    input_variables[0][batch_offset + output_offset + params.output_offset] = local_var
    vc.end()

    vc.set_global_builder(old_builder)

    return vd.ShaderObject(builder.build(name), signature, local_size=(group_size, 1, 1))
