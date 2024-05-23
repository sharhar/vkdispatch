import copy
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

import vkdispatch as vd


class ShaderDispatcher:
    """TODO: Docstring"""

    plan: vd.ComputePlan
    source: str
    pc_buff_dict: Dict[str, vd.PushConstantBuffer]
    my_local_size: Tuple[int, int, int]
    func_args: List[vd.ShaderVariable]

    def __init__(
        self,
        plan: vd.ComputePlan,
        source: str,
        pc_buff_dict: dict,
        my_local_size: Tuple[int, int, int],
        func_args: List[vd.ShaderVariable],
    ):
        self.plan = plan
        self.pc_buff_dict = copy.deepcopy(pc_buff_dict)
        self.my_local_size = my_local_size
        self.func_args = func_args
        self.source = source

    def __repr__(self) -> str:
        result = ""

        for ii, line in enumerate(self.source.split("\n")):
            result += f"{ii + 1:4d}: {line}\n"

        return result

    def __getitem__(self, exec_dims: Union[tuple, int]):
        my_blocks = [exec_dims, 1, 1]
        my_cmd_list: List[vd.CommandList] = [None]

        if isinstance(exec_dims, tuple):
            my_blocks = [1, 1, 1]

            for i, val in enumerate(exec_dims):
                if isinstance(val, int) or np.issubdtype(type(val), np.integer):
                    my_blocks[i] = val
                else:
                    if not isinstance(val, vd.CommandList) and val is not None:
                        raise ValueError(f"Invalid dimension '{val}'!")

                    if not i == len(exec_dims) - 1:
                        raise ValueError("Only the last dimension can be a command list!")

                    my_cmd_list[0] = val

        my_limits_x = my_blocks[0]
        my_limits_y = my_blocks[1]
        my_limits_z = my_blocks[2]

        my_blocks[0] = (my_blocks[0] + self.my_local_size[0] - 1) // self.my_local_size[0]
        my_blocks[1] = (my_blocks[1] + self.my_local_size[1] - 1) // self.my_local_size[1]
        my_blocks[2] = (my_blocks[2] + self.my_local_size[2] - 1) // self.my_local_size[2]

        def wrapper_func(*args, **kwargs):
            if len(args) != len(self.func_args):
                raise ValueError(
                    f"Expected {len(self.func_args)} arguments, got {len(args)}!"
                )

            descriptor_set = vd.DescriptorSet(self.plan._handle)
            pc_buff = vd.PushConstantBuffer(self.pc_buff_dict)

            pc_buff["exec_count"] = [my_limits_x, my_limits_y, my_limits_z, 0]

            for ii, arg in enumerate(self.func_args):
                descriptor_set.bind_buffer(args[ii], arg.binding)

            for key, val in kwargs.items():
                pc_buff[key] = val

            if len(kwargs) == len(pc_buff.pc_dict) - 1 and my_cmd_list[0] is None:
                cmd_list = vd.get_command_list()
                cmd_list.add_pc_buffer(pc_buff)
                cmd_list.add_desctiptor_set(descriptor_set)
                self.plan.record(cmd_list, descriptor_set, my_blocks)
                cmd_list.submit()
                return

            if my_cmd_list[0] is None:
                raise ValueError(
                    "Must provide all dynamic constants if no command list is specified!"
                )

            my_cmd_list[0].add_pc_buffer(pc_buff)
            my_cmd_list[0].add_desctiptor_set(descriptor_set)
            self.plan.record(my_cmd_list[0], descriptor_set, my_blocks)

            return pc_buff

        return wrapper_func


def compute_shader(*args, local_size: Tuple[int, int, int] = None):
    my_local_size = (
        local_size
        if local_size is not None
        else [vd.get_context().device_infos[0].max_workgroup_size[0], 1, 1]
    )

    def decorator(build_func):
        builder = vd.shader

        pc_exec_count_var = builder.push_constant(vd.uvec4, "exec_count")

        builder.if_statement(pc_exec_count_var[0] <= builder.global_x)
        builder.return_statement()
        builder.end()

        func_args = []

        for buff in args:
            if (isinstance(buff, vd.dtype)
                and buff.structure == vd.dtype_structure.DATA_STRUCTURE_BUFFER):
                func_args.append(builder.dynamic_buffer(buff))
            else:
                raise ValueError("Decorator must be given list of vd.dtype's only!")

        if len(func_args) > 0:
            build_func(*func_args)
        else:
            build_func()

        shader_source = builder.build(
            my_local_size[0], my_local_size[1], my_local_size[2]
        )

        plan = vd.ComputePlan(shader_source, builder.binding_count, builder.pc_size)

        wrapper = ShaderDispatcher(plan, shader_source, builder.pc_dict, my_local_size, func_args)

        builder.reset()

        return wrapper

    return decorator
