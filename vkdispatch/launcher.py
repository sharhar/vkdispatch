import vkdispatch as vd
import vkdispatch.codegen as vc

import copy

import typing

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable

import numpy as np

class LaunchBindObject:
    def __init__(self, parent: "LaunchVariables", name: str) -> None:
        self.parent = parent
        self.name = name
    
    def register_pc_var(self, buff_obj: vc.BufferStructureProxy, var_name: str):
        self.parent._register(self.name, buff_obj, var_name)

class LaunchVariables:
    def __init__(self) -> None:
        self.key_list = []
        self.pc_dict = {}

    def _register(self, name: str, buff_obj: vc.BufferStructureProxy, var_name: str):
        self.pc_dict[name] = (buff_obj, var_name)
    
    def __getitem__(self, name: str):
        if name in self.key_list:
            raise ValueError("Variable already bound!")
        
        self.key_list.append(name)

        return LaunchBindObject(self, name)
    
    def __setitem__(self, name: str, value):
        self.pc_dict[name][0][self.pc_dict[name][1]] = value

class LaunchParametersHolder:
    def __init__(self, func_args, args, kwargs) -> None:
        self.ref_dict = {}

        for ii, arg_var in enumerate(func_args):
            arg = None
            
            if ii < len(args):
                arg = args[ii]
            else:
                if arg_var[1] not in kwargs:
                    if arg_var[2] is None:
                        raise ValueError(f"Missing argument '{arg_var[1]}'!")
                    
                    arg = arg_var[2]
                else:
                    arg = kwargs[arg_var[1]]
            
            self.ref_dict[arg_var[1]] = arg

    def __getattr__(self, name: str):
        return self.ref_dict[name]

class ShaderLauncher:
    plan: vd.ComputePlan
    source: str
    pc_buff_dict: Dict[str, Tuple[int, vd.dtype]]
    uniform_buff_dict: Dict[str, Tuple[int, vd.dtype]]
    func_args: List[Tuple[vc.BaseVariable, str, typing.Any]]
    my_local_size: Tuple[int, int, int]
    my_workgroups: Tuple[int, int, int]
    my_exec_size: Tuple[int, int, int]

    def __init__(
        self,
        shader_source: str,
        pc_size: int,
        pc_buff_dict: dict,
        uniform_buff_dict: dict,
        binding_type_list: List[int],
        func_args: List[Tuple[vc.BaseVariable, str, typing.Any]],
        my_local_size: Tuple[int, int, int],
        my_workgroups: Tuple[int, int, int],
        my_exec_size: Tuple[int, int, int],
    ):
        self.plan = vd.ComputePlan(shader_source, binding_type_list, pc_size)
        self.pc_buff_dict = copy.deepcopy(pc_buff_dict)
        self.uniform_buff_dict = copy.deepcopy(uniform_buff_dict)
        self.func_args = func_args
        self.source = shader_source
        self.my_local_size = my_local_size
        self.my_workgroups = my_workgroups
        self.my_exec_size = my_exec_size

    def __repr__(self) -> str:
        result = ""

        for ii, line in enumerate(self.source.split("\n")):
            result += f"{ii + 1:4d}: {line}\n"

        return result

    def sanitize_dims_tuple(self, in_val, args, kwargs):
        if callable(in_val):
            in_val = in_val(LaunchParametersHolder(self.func_args, args, kwargs))

        if isinstance(in_val, int) or np.issubdtype(type(in_val), np.integer):
            return (in_val, 1, 1)

        if not isinstance(in_val, tuple):
            raise ValueError("Must provide a tuple of dimensions!")
        
        if len(in_val) > 0 and len(in_val) < 4:
            raise ValueError("Must provide a tuple of length 1, 2, or 3!")
        
        return_val = (1, 1, 1)

        for ii, val in enumerate(in_val):
            if not isinstance(val, int) and not np.issubdtype(type(val), np.integer):
                raise ValueError("All dimensions must be integers!")
            
            return_val[ii] = val
        
        return return_val

    def __call__(self, *args, **kwargs):
        my_blocks = None
        my_limits = None
        my_cmd_list = None

        if "workgroups" in kwargs or self.my_workgroups is not None:
            true_dims = self.sanitize_dims_tuple(
                kwargs["workgroups"]
                if "workgroups" in kwargs
                else self.my_workgroups,
                args, kwargs
            )

            my_blocks = true_dims
            my_limits = (true_dims[0] * self.my_local_size[0],
                         true_dims[1] * self.my_local_size[1],
                         true_dims[2] * self.my_local_size[2])
        
        if "exec_size" in kwargs or self.my_exec_size is not None:
            true_dims = self.sanitize_dims_tuple(
                kwargs["exec_size"]
                if "exec_size" in kwargs
                else self.my_exec_size,
                args, kwargs
            )

            my_limits = true_dims
            my_blocks = ((true_dims[0] + self.my_local_size[0] - 1) // self.my_local_size[0],
                         (true_dims[1] + self.my_local_size[1] - 1) // self.my_local_size[1],
                         (true_dims[2] + self.my_local_size[2] - 1) // self.my_local_size[2])

        if my_blocks is None:
            raise ValueError("Must provide either 'exec_size' or 'workgroups'!")
        
        if "cmd_list" in kwargs:
            my_cmd_list = kwargs["cmd_list"]
        
        if my_cmd_list is None:
            my_cmd_list = vd.get_command_list()
        
        descriptor_set = vd.DescriptorSet(self.plan._handle)

        pc_buff = None if self.pc_buff_dict is None else vc.BufferStructureProxy(self.pc_buff_dict, 0)
        static_constant_buffer = vc.BufferStructureProxy(self.uniform_buff_dict, vd.get_context().device_infos[0].uniform_buffer_alignment)

        static_constant_buffer["exec_count"] = [my_limits[0], my_limits[1], my_limits[2], 0]

        for ii, arg_var in enumerate(self.func_args):
            arg = None
            
            if ii < len(args):
                arg = args[ii]
            else:
                if arg_var[1] not in kwargs:
                    if arg_var[2] is None:
                        raise ValueError(f"Missing argument '{arg_var[1]}'!")
                    
                    arg = arg_var[2]
                else:
                    arg = kwargs[arg_var[1]]

            if isinstance(arg_var[0], vc.BufferVariable):
                if not isinstance(arg, vd.Buffer):
                    raise ValueError(f"Expected a buffer for argument '{arg_var[1]}'!")
                
                descriptor_set.bind_buffer(arg, arg_var[0].binding)
                static_constant_buffer[arg_var[0].shape_name] = arg.shader_shape

            elif isinstance(arg_var[0], vc.ImageVariable):
                if not isinstance(arg, vd.Image):
                    raise ValueError(f"Expected an image for argument '{arg_var[1]}'!")
                
                descriptor_set.bind_image(arg, arg_var[0].binding)
            
            elif isinstance(arg_var[0], vc.ShaderVariable):
                if arg_var[0].name.startswith("UBO."):
                    if isinstance(arg, LaunchBindObject):
                        raise ValueError("Cannot use LaunchVariables for Constants")

                    static_constant_buffer[arg_var[0].name[4:]] = arg
                elif arg_var[0].name.startswith("PC."):
                    if pc_buff is None:
                        raise ValueError("Something went wrong with push constants!!")
                    
                    if isinstance(arg, LaunchBindObject):
                        if my_cmd_list.submit_on_record:
                            raise ValueError("Cannot bind Variables for default cmd list!")

                        arg.register_pc_var(pc_buff, arg_var[0].name[3:])
                    else:
                        pc_buff[arg_var[0].name[3:]] = arg
                else:
                    raise ValueError(f"Unknown variable type '{arg_var[0].name}'!")
            else:
                raise ValueError(f"Something very wrong happened!")

        if pc_buff is not None:
            my_cmd_list.add_pc_buffer(pc_buff)
        my_cmd_list.add_desctiptor_set_and_static_constants(descriptor_set, static_constant_buffer)
        self.plan.record(my_cmd_list, descriptor_set, my_blocks)

        if my_cmd_list.submit_on_record:
            my_cmd_list.submit()
