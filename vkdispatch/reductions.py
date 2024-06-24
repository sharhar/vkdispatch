import vkdispatch as vd

import numpy as np
import typing

subgroup_operations = {
    "subgroupAdd": (vd.shader.subgroup_add, lambda x, y: x + y, 0),
    "subgroupMul": (vd.shader.subgroup_mul, lambda x, y: x * y, 1),
    "subgroupMin": (vd.shader.subgroup_min, vd.shader.min, np.inf),
    "subgroupMax": (vd.shader.subgroup_max, vd.shader.max, -np.inf),
    "subgroupAnd": (vd.shader.subgroup_and, lambda x, y: x & y, -1),
    "subgroupOr":  (vd.shader.subgroup_or,  lambda x, y: x | y, 0),
    "subgroupXor": (vd.shader.subgroup_xor, lambda x, y: x ^ y, 0),
}

class ReductionDispatcher:
    def __init__(
            self, 
            stage1: vd.ShaderDispatcher, 
            stage2: vd.ShaderDispatcher, 
            arg_count: int, 
            group_size: int,
            out_type: vd.dtype
        ) -> None:
        self.stage1 = stage1
        self.stage2 = stage2
        self.arg_count = arg_count
        self.group_size = group_size
        self.out_type = out_type

    def __repr__(self) -> str:
        return f"Stage 1:\n{self.stage1}\nStage 2:\n{self.stage2}"

    def __getitem__(self, exec_dims: typing.Union[tuple, int]):
        if isinstance(exec_dims, int):
            exec_dims = (exec_dims,)

        if len(exec_dims) < 1:
            raise ValueError("At least a size must be provided!")

        if len(exec_dims) > 2:
            raise ValueError("Only two arguments can be given for the execution dimensions!")
        
        if not isinstance(exec_dims[0], (int, np.integer)):
            raise ValueError("First excution dimention must be an int!")
        
        if len(exec_dims) == 2:
            if not isinstance(exec_dims[1], vd.CommandList) and exec_dims[1] is not None:
                raise ValueError("Second excution dimention must be a CommandList!")
            
        if len(exec_dims) == 1:
            exec_dims = (exec_dims[0], None)

        my_exec_dims = [None]
        my_exec_dims[0] = exec_dims

        def wrapper_func(*args, **kwargs):
            if len(args) != self.arg_count:
                raise ValueError(f"Expected {self.arg_count} arguments, got {len(args)}!")
            
            data_size = my_exec_dims[0][0]
            stage1_blocks = int(np.ceil(data_size / self.group_size))
            
            if stage1_blocks > self.group_size:
                reduction_scale = int(np.ceil(stage1_blocks / self.group_size))
                stage_scale = np.ceil(np.sqrt(reduction_scale))
                stage1_blocks = int(np.ceil(stage1_blocks / stage_scale))

            reduction_buffer = vd.Buffer((stage1_blocks + 1,), self.out_type)

            self.stage1[stage1_blocks * self.group_size, my_exec_dims[0][1]](reduction_buffer, *args, N=data_size)
            self.stage2[self.group_size, my_exec_dims[0][1]](reduction_buffer, N=stage1_blocks+1)
            
            return reduction_buffer

        return wrapper_func

def make_reduction(
        reduce: typing.Union[
                    typing.Callable[[vd.ShaderVariable, vd.ShaderVariable], vd.ShaderVariable],
                    str
                ],
        out_type: vd.dtype,
        *var_types: vd.dtype,
        reduction_identity = None,
        group_size: int = None,
        map: typing.Callable[[typing.List[vd.ShaderVariable]], vd.ShaderVariable] = None):
    if len(var_types) == 0:
        raise ValueError("At least one buffer type must be provided!")
    
    if len(var_types) > 1 and map is None:
        raise ValueError("A map function must be provided for multiple buffer types!")
    
    subgroup_size = vd.get_context().device_infos[0].sub_group_size
    
    if group_size is None:
        group_size = vd.get_context().device_infos[0].max_workgroup_size[0]
    
    if group_size % subgroup_size != 0:
        raise ValueError("Group size must be a multiple of the sub-group size!")
    
    if isinstance(reduce, str) and reduce not in subgroup_operations.keys():
            raise ValueError("Invalid reduction operation!")
    
    if reduction_identity is None:
        reduction_identity = subgroup_operations[reduce][2] if isinstance(reduce, str) else 0

    reduction_func = subgroup_operations[reduce][1] if isinstance(reduce, str) else reduce

    def create_reduction_stage(reduction_map, first_input_index, in_var_types):
        @vd.compute_shader(*in_var_types, local_size=(group_size, 1, 1))
        def reduction_stage(*buffers):
            ind = vd.shader.global_x.copy(var_name="ind")

            N = vd.shader.push_constant(vd.int32, "N")
            offset = vd.shader.new(vd.uint32, 1 - first_input_index, var_name="offset")

            reduction_aggregate = vd.shader.new(out_type, reduction_identity, var_name="reduction_aggregate")

            vd.shader.while_statement(ind + offset < N)

            if reduction_map is not None:
                mapped_inputs = [buffer[ind + offset] for buffer in buffers[1:]]
                reduction_aggregate[:] = reduction_func(reduction_aggregate, reduction_map(*mapped_inputs).copy())
            else:
                reduction_aggregate[:] = reduction_func(reduction_aggregate, buffers[first_input_index][ind + offset])
            offset += vd.shader.workgroup_size_x * vd.shader.num_workgroups_x
            vd.shader.end()

            tid = vd.shader.local_x.copy(var_name="tid")
        
            sdata = vd.shader.shared_buffer(out_type, group_size, var_name="sdata")

            sdata[tid] = reduction_aggregate

            vd.shader.memory_barrier_shared()
            vd.shader.barrier()
            
            current_size = group_size // 2
            while current_size > subgroup_size:
                vd.shader.if_statement(tid < current_size)
                sdata[tid] = reduction_func(sdata[tid], sdata[tid + current_size])            
                vd.shader.end()
                
                vd.shader.memory_barrier_shared()
                vd.shader.barrier()
                
                current_size //= 2
            
            if group_size > subgroup_size:
                vd.shader.if_statement(tid < subgroup_size)
                sdata[tid] = reduction_func(sdata[tid], sdata[tid + subgroup_size])
                vd.shader.end()
                vd.shader.subgroup_barrier()
            
            if isinstance(reduce, str):
                local_var = sdata[tid].copy(var_name="local_var")
                local_var[:] = subgroup_operations[reduce][0](local_var)

                vd.shader.if_statement(tid == 0)
                buffers[0][vd.shader.workgroup_x + first_input_index] = local_var
                vd.shader.end()
            else:
                current_size = subgroup_size // 2
                while current_size > 1:
                    vd.shader.if_statement(tid < current_size)
                    sdata[tid] = reduction_func(sdata[tid], sdata[tid + current_size])
                    vd.shader.end()
                    
                    vd.shader.subgroup_barrier()
                    
                    current_size //= 2
                
                vd.shader.if_statement(tid == 0)
                buffers[0][vd.shader.workgroup_x + first_input_index] = reduction_func(sdata[tid], sdata[tid + current_size])
                vd.shader.end()

        return reduction_stage

    return ReductionDispatcher(
        create_reduction_stage(map, 1, (out_type[0], *var_types)),
        create_reduction_stage(None, 0, (out_type[0], )),
        len(var_types),
        group_size,
        out_type
    )
    
def map_reduce(
        out_type, 
        *args, 
        group_size: int = None,
        reduction: typing.Union[
                    typing.Callable[[vd.ShaderVariable, vd.ShaderVariable], vd.ShaderVariable],
                    str
                ] = None,
        reduction_identity = None, 
        map: typing.Callable[[typing.List[vd.ShaderVariable]], vd.ShaderVariable] = None):
    def decorator(build_func):
        my_map = map
        my_reduction = reduction
        
        if my_reduction is None:
            my_reduction = build_func
        elif my_map is None:
            my_map = build_func

        return make_reduction(my_reduction, out_type, *args, group_size=group_size, map=my_map, reduction_identity=reduction_identity)

    return decorator