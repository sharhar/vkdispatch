import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import List
from typing import Tuple
from typing import Optional
from typing import Callable
from typing import Union

import numpy as np
import inspect

subgroup_operations = {
    "subgroupAdd": (vc.subgroup_add, lambda x, y: x + y, 0),
    "subgroupMul": (vc.subgroup_mul, lambda x, y: x * y, 1),
    "subgroupMin": (vc.subgroup_min, vc.min, np.inf),
    "subgroupMax": (vc.subgroup_max, vc.max, -np.inf),
    "subgroupAnd": (vc.subgroup_and, lambda x, y: x & y, -1),
    "subgroupOr":  (vc.subgroup_or,  lambda x, y: x | y, 0),
    "subgroupXor": (vc.subgroup_xor, lambda x, y: x ^ y, 0),
}

def create_reduction_stage(reduction_map, first_input_index, stage_signature, out_type, reduce, reduction_identity, group_size, subgroup_size):
    if isinstance(reduce, str) and reduce not in subgroup_operations.keys():
            raise ValueError("Invalid reduction operation!")
    
    if reduction_identity is None:
        reduction_identity = subgroup_operations[reduce][2] if isinstance(reduce, str) else 0

    reduction_func = subgroup_operations[reduce][1] if isinstance(reduce, str) else reduce

    @vd.shader(local_size=(group_size, 1, 1), annotations=stage_signature)
    def reduction_stage(*in_vars):
        ind = vc.global_invocation().x.copy("ind")
        
        offset = vc.new(vd.uint32, 1 - first_input_index, var_name="offset")

        reduction_aggregate = vc.new(out_type, reduction_identity, var_name="reduction_aggregate")

        N = in_vars[-1].copy("N")
        buffers = in_vars[:-1]

        vc.while_statement(ind + offset < N)

        if reduction_map is not None:
            reduction_aggregate[:] = reduction_func(reduction_aggregate, reduction_map(ind + offset, *buffers[1:]).copy("mapped_value"))
        else:
            reduction_aggregate[:] = reduction_func(reduction_aggregate, buffers[first_input_index][ind + offset])
        offset += vc.workgroup_size().x * vc.num_workgroups().x
        vc.end()

        tid = vc.local_invocation().x.copy("tid")
    
        sdata = vc.shared_buffer(out_type, group_size, var_name="sdata")

        sdata[tid] = reduction_aggregate

        vc.memory_barrier_shared()
        vc.barrier()
        
        current_size = group_size // 2
        while current_size > subgroup_size:
            vc.if_statement(tid < current_size)
            sdata[tid] = reduction_func(sdata[tid], sdata[tid + current_size])            
            vc.end()
            
            vc.memory_barrier_shared()
            vc.barrier()
            
            current_size //= 2
        
        if group_size > subgroup_size:
            vc.if_statement(tid < subgroup_size)
            sdata[tid] = reduction_func(sdata[tid], sdata[tid + subgroup_size])
            vc.end()
            vc.subgroup_barrier()
        
        if isinstance(reduce, str):
            local_var = sdata[tid].copy()
            local_var[:] = subgroup_operations[reduce][0](local_var)

            vc.if_statement(tid == 0)
            buffers[0][vc.workgroup().x + first_input_index] = local_var
            vc.end()
        else:
            current_size = subgroup_size // 2
            while current_size > 1:
                vc.if_statement(tid < current_size)
                sdata[tid] = reduction_func(sdata[tid], sdata[tid + current_size])
                vc.end()
                
                vc.subgroup_barrier()
                
                current_size //= 2
            
            vc.if_statement(tid == 0)
            buffers[0][vc.workgroup().x + first_input_index] = reduction_func(sdata[tid], sdata[tid + current_size])
            vc.end()

    return reduction_stage

class ReductionDispatcher:
    def __init__(self, reduce: Union[
                            Callable[[vc.ShaderVariable, vc.ShaderVariable], vc.ShaderVariable],
                            str
                        ],
                out_type: vd.dtype,
                *var_types: vd.dtype,
                reduction_identity = None,
                group_size: Optional[int] = None,
                map: Optional[Callable[[List[vc.ShaderVariable]], vc.ShaderVariable]] = None,
                exec_size = None,
                func_args = None):
        if len(var_types) == 0:
            raise ValueError("At least one buffer type must be provided!")
        
        if len(var_types) > 1 and map is None:
            raise ValueError("A map function must be provided for multiple buffer types!")
        
        self.stage1_args = (map, 1, (vc.Buffer[out_type], *var_types[1:], vc.Const[vc.i32]), out_type, reduce, reduction_identity)
        self.stage2_args = (None, 0, (vc.Buffer[out_type], vc.Const[vc.i32]), out_type, reduce, reduction_identity)

        self.stage1 = None
        self.stage2 = None

        #self.stage1 = create_reduction_stage(map, 1, (vc.Buffer[out_type], *var_types[1:], vc.Const[vc.i32]), out_type, reduce, reduction_identity, group_size, subgroup_size)
        #self.stage2 = create_reduction_stage(None, 0, (vc.Buffer[out_type], vc.Const[vc.i32]), out_type, reduce, reduction_identity, group_size, subgroup_size)
        self.arg_count = len(var_types[1:])
        self.group_size = group_size
        self.out_type = out_type
        self.exec_size = exec_size
        self.func_args = func_args
    
    def make_stages(self):
        if self.stage1 is not None and self.stage2 is not None:
            return

        subgroup_size = vd.get_context().subgroup_size
        
        if self.group_size is None:
            self.group_size = vd.get_context().max_workgroup_size[0]
        
        if self.group_size % subgroup_size != 0:
            raise ValueError("Group size must be a multiple of the sub-group size!")
        
        self.stage1 = create_reduction_stage(*self.stage1_args, self.group_size, subgroup_size)
        self.stage2 = create_reduction_stage(*self.stage2_args, self.group_size, subgroup_size)

    def __repr__(self) -> str:
        self.make_stages()

        return f"Stage 1:\n{self.stage1}\nStage 2:\n{self.stage2}"
    
    def __call__(self, *args, **kwargs) -> vd.Buffer:
        self.make_stages()

        my_cmd_stream = None

        my_exec_size = self.exec_size

        if "exec_size" in kwargs:
            my_exec_size = kwargs["exec_size"]

        if my_exec_size is None:
            raise ValueError("Execution size must be provided!")

        if callable(my_exec_size):

            actual_arg_list = [
                (arg[1], arg[2]) for arg in self.func_args[1:]
            ]

            my_exec_size = my_exec_size(vd.LaunchParametersHolder(actual_arg_list, args, kwargs))
        
        if not isinstance(my_exec_size, int) and not np.issubdtype(type(my_exec_size), np.integer):
            raise ValueError("Execution size must be an integer!")
        
        my_exec_size = my_exec_size
        
        if "cmd_stream" in kwargs:
            my_cmd_stream = kwargs["cmd_stream"]
        
        if my_cmd_stream is None:
            my_cmd_stream = vd.global_cmd_stream()

        stage1_blocks = int(np.ceil(my_exec_size / self.group_size))
        
        if stage1_blocks > self.group_size:
            reduction_scale = int(np.ceil(stage1_blocks / self.group_size))
            stage_scale = np.ceil(np.sqrt(reduction_scale))
            stage1_blocks = int(np.ceil(stage1_blocks / stage_scale))

        reduction_buffer = vd.Buffer((stage1_blocks + 1,), self.out_type)

        self.stage1(reduction_buffer, *args, my_exec_size, exec_size=stage1_blocks * self.group_size, cmd_stream=my_cmd_stream)
        self.stage2(reduction_buffer, stage1_blocks+1, exec_size=self.group_size, cmd_stream=my_cmd_stream)

        return reduction_buffer
    
def map_reduce(
        exec_size = None,
        group_size: int = None,
        reduction: Optional[Union[
                    Callable[[vc.ShaderVariable, vc.ShaderVariable], vc.ShaderVariable],
                    str
                ]] = None,
        reduction_identity = None, 
        map: Optional[Callable[[List[vc.ShaderVariable]], vc.ShaderVariable]] = None,
        signature: tuple = None):
    def decorator(build_func):
        func_signature = inspect.signature(build_func)
        
        my_map = map
        my_reduction = reduction
        
        if my_reduction is None:
            my_reduction = build_func
        elif my_map is None:
            my_map = build_func

        return_annotation = func_signature.return_annotation

        if return_annotation == inspect.Parameter.empty:
            raise ValueError("Return type must be annotated")

        mapping_signature = []
        func_args = []

        for ii, param in enumerate(func_signature.parameters.values()):
            my_annotation = param.annotation if signature is None else signature[ii]

            if my_annotation == inspect.Parameter.empty:
                raise ValueError("All parameters must be annotated")

            if not hasattr(my_annotation, '__args__'):
                raise TypeError(f"Argument '{param.name}: vd.{my_annotation}' must have a type annotation")

            mapping_signature.append(my_annotation)

            func_args.append((None, param.name, 
                              param.default if param.default != inspect.Parameter.empty else None
                            ))
        
        return ReductionDispatcher(
            my_reduction, 
            return_annotation, 
            *mapping_signature, 
            group_size=group_size, 
            map=my_map, 
            reduction_identity=reduction_identity, 
            exec_size=exec_size,
            func_args=func_args
        )

    return decorator