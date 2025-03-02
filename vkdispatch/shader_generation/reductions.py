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
    "subgroupAdd": vd.SubgroupAdd,
    "subgroupMul": vd.SubgroupMul,
    "subgroupMin": vd.SubgroupMin,
    "subgroupMax": vd.SubgroupMax,
    "subgroupAnd": vd.SubgroupAnd,
    "subgroupOr":  vd.SubgroupOr,
    "subgroupXor": vd.SubgroupXor,
}

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
        
        if isinstance(reduce, str):
            self.reduction = subgroup_operations[reduce]
        else:
            self.reduction = vd.ReductionOperation(
                name="custom",
                reduction=reduce,
                identity=reduction_identity
            )

        self.map_func = map
        self.out_type = out_type

        self.stage1_input_types = var_types[1:]

        self.stage1 = None
        self.stage2 = None

        self.arg_count = len(var_types[1:])
        self.group_size = group_size
        self.out_type = out_type
        self.exec_size = exec_size
        self.func_args = func_args
    
    def make_stages(self):
        if self.stage1 is not None and self.stage2 is not None:
            return
        
        if self.group_size is None:
            self.group_size = vd.get_context().max_workgroup_size[0]
        
        if self.group_size % vd.get_context().subgroup_size != 0:
            raise ValueError("Group size must be a multiple of the sub-group size!")
        
        self.stage1 = vd.make_reduction_stage(
            self.reduction, 
            self.out_type, 
            self.group_size, 
            False, 
            map_func=self.map_func, 
            input_types=self.stage1_input_types
        )

        self.stage2 = vd.make_reduction_stage(
            self.reduction, 
            self.out_type, 
            self.group_size, 
            True,
        )

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

        stage1_params = vd.ReductionParams(
            input_offset=0,
            input_size=my_exec_size,
            input_stride=1,
            input_batch_stride=0,
            output_offset=1,
            output_stride=1,
            output_batch_stride=0,
        )

        self.stage1(reduction_buffer, *args, stage1_params, exec_size=stage1_blocks * self.group_size, cmd_stream=my_cmd_stream)

        stage2_params = vd.ReductionParams(
            input_offset=1,
            input_size=stage1_blocks+1,
            input_stride=1,
            input_batch_stride=0,
            output_offset=0,
            output_stride=1,
            output_batch_stride=0,
        )

        self.stage2(reduction_buffer, stage2_params, exec_size=self.group_size, cmd_stream=my_cmd_stream)

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