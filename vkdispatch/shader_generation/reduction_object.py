import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Callable
from typing import List

import numpy as np

class ReductionObject:
    def __init__(self,
                 reduction: vd.ReductionOperation,
                 out_type: vd.dtype, 
                 group_size: int = None, 
                 map_func: Callable = None,
                 input_types: List = None,
                 axes = None):
        self.reduction = reduction
        self.out_type = out_type
        self.group_size = group_size
        self.map_func = map_func
        self.input_types = input_types if input_types is not None else [vc.Buffer[out_type]]

        assert axes is None, "Axes are not yet supported!"

        self.stage1 = None
        self.stage2 = None

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
            input_types=self.input_types
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

        """
        my_exec_size = self.exec_size

        if "exec_size" in kwargs:
            my_exec_size = kwargs["exec_size"]

        if my_exec_size is None:
            raise ValueError("Execution size must be provided!")

        if callable(my_exec_size):

            actual_arg_list = [
                (arg[1], arg[2]) for arg in self.input_types
            ]

            my_exec_size = my_exec_size(vd.LaunchParametersHolder(actual_arg_list, args, kwargs))
        
        if not isinstance(my_exec_size, int) and not np.issubdtype(type(my_exec_size), np.integer):
            raise ValueError("Execution size must be an integer!")
        
        my_exec_size = my_exec_size
        """

        my_exec_size = args[0].size
        
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

        self.stage1(reduction_buffer, *args, stage1_params, exec_size=my_exec_size, cmd_stream=my_cmd_stream)

        stage2_params = vd.ReductionParams(
            input_offset=1,
            input_size=stage1_blocks+1,
            input_stride=1,
            input_batch_stride=0,
            output_offset=0,
            output_stride=1,
            output_batch_stride=0,
        )

        self.stage2(reduction_buffer, stage2_params, exec_size=stage1_blocks, cmd_stream=my_cmd_stream)

        return reduction_buffer