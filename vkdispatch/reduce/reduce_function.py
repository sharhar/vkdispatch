import vkdispatch as vd
import vkdispatch.codegen as vc

from .operations import ReduceOp
from .stage import make_reduction_stage, ReductionParams

from typing import List, Optional

from .._compat import numpy_compat as npc

class ReduceFunction:
    def __init__(self,
                 reduction: ReduceOp,
                 group_size: int = None, 
                 axes: List[int] = None,
                 mapping_function: Optional[vd.MappingFunction] = None):
        self.reduction = reduction
        self.out_type = mapping_function.return_type
        self.group_size = group_size
        self.map_func = mapping_function
        self.input_types = mapping_function.buffer_types
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
        
        self.stage1 = make_reduction_stage(
            self.reduction, 
            self.out_type, 
            self.group_size, 
            False, 
            map_func=self.map_func, 
            input_types=self.input_types
        )

        self.stage2 = make_reduction_stage(
            self.reduction, 
            self.out_type, 
            self.group_size, 
            True,
        )

    def get_src(self, line_numbers: bool = None) -> str:
        self.make_stages()

        return [
            self.stage1.get_src(line_numbers),
            self.stage2.get_src(line_numbers)
        ]
    
    def print_src(self, line_numbers: bool = None):
        srcs = self.get_src(line_numbers)

        print(f"// Reduction Stage 1:\n{srcs[0]}\n// Reduction Stage 2:\n{srcs[1]}")
    
    def __repr__(self) -> str:
        self.make_stages()

        srcs = self.get_src()

        return f"// Reduction Stage 1:\n{srcs[0]}\n// Reduction Stage 2:\n{srcs[1]}"

    def __call__(self, *args, **kwargs) -> vd.Buffer:
        self.make_stages()

        my_graph = kwargs.get("graph", vd.global_graph())

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

        workgroups_x = int(npc.ceil(input_size / (self.group_size * input_stride)))

        if workgroups_x > self.group_size:
            workgroups_x = self.group_size

        output_buffer_shape = [workgroups_x + 1]

        if batch_count > 1:
            output_buffer_shape.append(batch_count)
        
        if input_stride > 1:
            output_buffer_shape.append(input_stride)

        reduction_buffer = vd.Buffer(tuple(output_buffer_shape), self.out_type)

        stage1_params = ReductionParams(
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

        self.stage1(reduction_buffer, *args, stage1_params, exec_size=stage1_exec_size, graph=my_graph)

        stage2_params = ReductionParams(
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

        self.stage2(reduction_buffer, stage2_params, exec_size=stage2_exec_size, graph=my_graph)

        return reduction_buffer
