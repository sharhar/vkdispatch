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
                 axes: List[int] = None):
        self.reduction = reduction
        self.out_type = out_type
        self.group_size = group_size
        self.map_func = map_func
        self.input_types = input_types if input_types is not None else [vc.Buffer[out_type]]
        self.axes = axes

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

        my_cmd_stream = kwargs.get("cmd_stream", vd.global_cmd_stream())

        input_size = 1
        input_stride = 1

        batch_count = 1

        if self.axes is None:
            input_size = args[0].size
        else:
            batched_axes = []

            for i in range(len(args[0].shape)):
                if i in self.axes:
                    break
                batched_axes.append(i)

            skipped_axes = []

            for i in range(len(args[0].shape) - 1, -1, -1):
                if i in self.axes:
                    break
                
                skipped_axes.append(i)


            for i in range(len(args[0].shape)):
                if i in batched_axes:
                    continue

                input_size *= args[0].shape[i]

            for i in batched_axes:
                batch_count *= args[0].shape[i]

            for i in skipped_axes:
                input_stride *= args[0].shape[i]

            assert input_stride == 1, "Reduction axes must be contiguous!"

        workgroups_x = int(np.ceil(input_size / (self.group_size * input_stride)))

        if workgroups_x > self.group_size:
            workgroups_x = self.group_size

        output_buffer_shape = [workgroups_x + 1]

        if batch_count > 1:
            output_buffer_shape.append(batch_count)
        
        if input_stride > 1:
            output_buffer_shape.append(input_stride)

        reduction_buffer = vd.Buffer(tuple(output_buffer_shape), self.out_type)

        print(reduction_buffer.shape)

        stage1_params = vd.ReductionParams(
            input_offset=0,
            input_size=input_size,
            input_stride=input_stride,
            input_y_batch_stride=input_size,
            input_z_batch_stride=0,
            output_offset=batch_count,
            output_stride=1,
            output_y_batch_stride=workgroups_x,
            output_z_batch_stride=0,
        )

        stage1_exec_size = (workgroups_x * self.group_size, batch_count)

        self.stage1(reduction_buffer, *args, stage1_params, exec_size=stage1_exec_size, cmd_stream=my_cmd_stream)

        stage2_params = vd.ReductionParams(
            input_offset=batch_count,
            input_size=workgroups_x,
            input_stride=1,
            input_y_batch_stride=workgroups_x,
            input_z_batch_stride=0,
            output_offset=0,
            output_stride=1,
            output_y_batch_stride=1,
            output_z_batch_stride=0,
        )

        stage2_exec_size = (self.group_size, batch_count)

        self.stage2(reduction_buffer, stage2_params, exec_size=stage2_exec_size, cmd_stream=my_cmd_stream)

        return reduction_buffer