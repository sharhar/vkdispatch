import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np
import typing
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

class ReductionDispatcher:
    def __init__(
            self, 
            stage1: "vd.ShaderLauncher", 
            stage2: "vd.ShaderLauncher", 
            arg_count: int, 
            group_size: int,
            out_type: vd.dtype,
            exec_size = None,
            func_args = None
        ) -> None:
        self.stage1 = stage1
        self.stage2 = stage2
        self.arg_count = arg_count
        self.group_size = group_size
        self.out_type = out_type
        self.exec_size = exec_size
        self.func_args = func_args

    def __repr__(self) -> str:
        return f"Stage 1:\n{self.stage1}\nStage 2:\n{self.stage2}"
    
    def __call__(self, *args, **kwargs) -> vd.Buffer:
        my_blocks = None
        my_limits = None
        my_cmd_list = None

        if "exec_size" in kwargs or self.exec_size is not None:
            true_dims = vd.sanitize_dims_tuple(
                self.func_args,
                kwargs["exec_size"]
                if "exec_size" in kwargs
                else self.exec_size,
                (None, *args), kwargs
            )

            my_limits = true_dims
            my_blocks = ((true_dims[0] + self.group_size - 1) // self.group_size,
                         1,
                         1)

        if my_blocks is None:
            raise ValueError("Must provide 'exec_size'!")
        
        if "cmd_list" in kwargs:
            my_cmd_list = kwargs["cmd_list"]
        
        if my_cmd_list is None:
            my_cmd_list = vd.get_command_list()

        data_size = my_limits[0]
        stage1_blocks = int(np.ceil(data_size / self.group_size))
        
        if stage1_blocks > self.group_size:
            reduction_scale = int(np.ceil(stage1_blocks / self.group_size))
            stage_scale = np.ceil(np.sqrt(reduction_scale))
            stage1_blocks = int(np.ceil(stage1_blocks / stage_scale))

        reduction_buffer = vd.Buffer((stage1_blocks + 1,), self.out_type)

        self.stage1(reduction_buffer, *args, data_size, exec_size=stage1_blocks * self.group_size, cmd_list=my_cmd_list)
        self.stage2(reduction_buffer, stage1_blocks+1, exec_size=self.group_size, cmd_list=my_cmd_list)

        return reduction_buffer

def make_reduction(
        reduce: typing.Union[
                    typing.Callable[[vc.ShaderVariable, vc.ShaderVariable], vc.ShaderVariable],
                    str
                ],
        out_type: vd.dtype,
        *var_types: vd.dtype,
        reduction_identity = None,
        group_size: typing.Union[int, None] = None,
        map: typing.Union[typing.Callable[[typing.List[vc.ShaderVariable]], vc.ShaderVariable], None] = None,
        exec_size = None,
        func_args = None):
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

    print("REDUCTION FUNC", reduction_func)

    def create_reduction_stage(reduction_map, first_input_index, stage_signature):
        @vc.shader(local_size=(group_size, 1, 1), signature=stage_signature)
        def reduction_stage(*in_vars): #, N: vc.Const[vc.i32]):
            ind = vc.global_invocation.x.copy()
            
            offset = vc.new(vd.uint32, 1 - first_input_index)

            reduction_aggregate = vc.new(out_type, reduction_identity)

            N = in_vars[-1]
            buffers = in_vars[:-1]

            vc.while_statement(ind + offset < N)

            if reduction_map is not None:
                reduction_aggregate[:] = reduction_func(reduction_aggregate, reduction_map(ind + offset, *buffers[1:]).copy())
            else:
                reduction_aggregate[:] = reduction_func(reduction_aggregate, buffers[first_input_index][ind + offset])
            offset += vc.workgroup_size.x * vc.num_workgroups.x
            vc.end()

            tid = vc.local_invocation.x.copy()
        
            sdata = vc.shared_buffer(out_type, group_size)

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
                buffers[0][vc.workgroup.x + first_input_index] = local_var
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
                buffers[0][vc.workgroup.x + first_input_index] = reduction_func(sdata[tid], sdata[tid + current_size])
                vc.end()

        return reduction_stage

    return ReductionDispatcher(
        create_reduction_stage(map, 1, (vc.Buffer[out_type], *var_types[1:], vc.Const[vc.i32])),
        create_reduction_stage(None, 0, (vc.Buffer[out_type], vc.Const[vc.i32])),
        len(var_types[1:]),
        group_size,
        out_type,
        exec_size,
        func_args
    )
    
def map_reduce(
        exec_size = None,
        group_size: int = None,
        reduction: typing.Union[
                    typing.Callable[[vc.ShaderVariable, vc.ShaderVariable], vc.ShaderVariable],
                    str
                ] = None,
        reduction_identity = None, 
        map: typing.Callable[[typing.List[vc.ShaderVariable]], vc.ShaderVariable] = None,
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
        
        return make_reduction(
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