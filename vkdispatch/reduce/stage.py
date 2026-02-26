import vkdispatch as vd
import vkdispatch.codegen as vc
from typing import List, Optional

from .operations import ReduceOp

import dataclasses

@dataclasses.dataclass
class ReductionParams:
    input_offset: vd.uint32
    input_size: vd.uint32
    input_stride: vd.uint32
    input_y_batch_stride: vd.uint32
    input_z_batch_stride: vd.uint32

    output_offset: vd.uint32
    output_stride: vd.uint32
    output_y_batch_stride: vd.uint32
    output_z_batch_stride: vd.uint32

__static_global_io_index: vc.ShaderVariable = None

def set_mapped_io_index(io_index: vc.ShaderVariable):
    global __static_global_io_index
    __static_global_io_index = io_index

def mapped_io_index() -> vc.ShaderVariable:
    return __static_global_io_index

def global_reduce(
        reduction: ReduceOp, 
        out_type: vd.dtype, 
        buffers: List[vc.BufferVariable], 
        params: ReductionParams,
        map_func: Optional[vd.MappingFunction] = None):
    
    ind = (vc.global_invocation_id().x * params.input_stride).to_register("ind")
    reduction_aggregate = vc.new_register(out_type, reduction.identity, var_name="reduction_aggregate")

    batch_offset = vc.workgroup_id().y * params.input_y_batch_stride
    inside_batch_offset = vc.workgroup_id().z * params.input_z_batch_stride

    start_index = vc.new_uint_register(params.input_offset + inside_batch_offset + batch_offset, var_name="start_index")

    current_index = vc.new_uint_register(start_index + ind, var_name="current_index")

    end_index = vc.new_uint_register(start_index + params.input_size, var_name="end_index")

    vc.while_statement(current_index < end_index)

    mapped_value = buffers[0][current_index]

    if map_func is not None:
        set_mapped_io_index(current_index)
        mapped_value = map_func.callback(*buffers)
        set_mapped_io_index(None)

    reduction_aggregate[:] = reduction.reduction(reduction_aggregate, mapped_value)

    current_index += vc.workgroup_size().x * vc.num_workgroups().x

    vc.end()

    return reduction_aggregate

def workgroup_reduce(
        reduction_aggregate: vc.ShaderVariable,
        reduction: ReduceOp,
        out_type: vd.dtype,
        group_size: int):
    tid = vc.local_invocation_id().x
    
    sdata = vc.shared_buffer(out_type, group_size, var_name="sdata")

    sdata[tid] = reduction_aggregate

    vc.barrier()
    
    current_size = group_size // 2
    while current_size > vd.get_context().subgroup_size:
        vc.if_statement(tid < current_size)
        sdata[tid] = reduction.reduction(sdata[tid], sdata[tid + current_size])            
        if current_size // 2 > vd.get_context().subgroup_size:
            vc.end()
        else:
            tid_limit = 2

            if vd.get_context().subgroup_size != 1:
                tid_limit = 2*vc.subgroup_size()

            vc.else_if_statement(tid < tid_limit)
            sdata[tid] = vc.new_register(out_type, 0)
            vc.end()
        
        vc.barrier()
        
        current_size //= 2

    return sdata

def subgroup_reduce(
        sdata: vc.ShaderVariable,
        reduction: ReduceOp,
        group_size: int):
    tid = vc.local_invocation_id().x
    subgroup_size = vd.get_context().subgroup_size

    if group_size > subgroup_size:
        vc.if_statement(tid < subgroup_size)
        sdata[tid] = reduction.reduction(sdata[tid], sdata[tid + subgroup_size])
        vc.end()

        if subgroup_size == 1:
            return sdata[tid].to_register("local_var")

        vc.subgroup_barrier()
    
    if reduction.subgroup_reduction is not None:
        local_var = sdata[tid].to_register("local_var")
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
        reduction: ReduceOp, 
        out_type: vd.dtype, 
        group_size: int, 
        output_is_input: bool,
        map_func: Optional[vd.MappingFunction] = None,
        input_types: List = None) -> vd.ShaderFunction:

    name = f"reduction_stage_{reduction.name}_{out_type.name}_{input_types}_{group_size}"
    
    with vd.shader_context() as context:
        signature_type_array = []
        
        signature_type_array.append(vc.Buffer[out_type])

        if input_types is not None:
            signature_type_array.extend(input_types)

        signature_type_array.append(ReductionParams)

        input_variables = context.declare_input_arguments(signature_type_array)

        params: ReductionParams = input_variables[-1]

        input_buffers = input_variables[:-1] if output_is_input else input_variables[1:-1]

        reduction_aggregate = global_reduce(reduction, out_type, input_buffers, params, map_func)
        sdata = workgroup_reduce(reduction_aggregate, reduction, out_type, group_size)
        local_var = subgroup_reduce(sdata, reduction, group_size)

        batch_offset = vc.workgroup_id().y * params.output_y_batch_stride
        output_offset = vc.workgroup_id().x * params.output_stride

        vc.if_statement(vc.local_invocation_id().x == 0)
        input_variables[0][batch_offset + output_offset + params.output_offset] = local_var
        vc.end()

        return context.get_function(local_size=(group_size, 1, 1), name=name)
