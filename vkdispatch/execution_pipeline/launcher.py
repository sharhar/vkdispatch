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

def sanitize_dims_tuple(func_args, in_val, args, kwargs) -> Tuple[int, int, int]:
    if callable(in_val):
        in_val = in_val(LaunchParametersHolder(func_args, args, kwargs))

    if isinstance(in_val, int) or np.issubdtype(type(in_val), np.integer):
        return (in_val, 1, 1) # type: ignore

    if not isinstance(in_val, tuple):
        raise ValueError("Must provide a tuple of dimensions!")
    
    if len(in_val) > 0 and len(in_val) < 4:
        raise ValueError("Must provide a tuple of length 1, 2, or 3!")
    
    return_val = [1, 1, 1]

    for ii, val in enumerate(in_val):
        if not isinstance(val, int) and not np.issubdtype(type(val), np.integer):
            raise ValueError("All dimensions must be integers!")
        
        return_val[ii] = val
    
    return (return_val[0], return_val[1], return_val[2])

class LaunchBindObject:
    def __init__(self, parent: "LaunchVariables", name: str) -> None:
        self.parent = parent
        self.name = name
    
    def register_pc_var(self, buff_obj: "vc.BufferStructureProxy", var_name: str):
        self.parent._register(self.name, buff_obj, var_name)

class LaunchVariables:
    def __init__(self) -> None:
        self.key_list = []
        self.pc_dict = {}

    def _register(self, name: str, buff_obj: "vc.BufferStructureProxy", var_name: str):
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
    shader_description: vc.ShaderDescription
    func_args: List[Tuple[vc.BaseVariable, str, typing.Any]]
    my_local_size: Tuple[int, int, int]
    my_workgroups: Tuple[int, int, int]
    my_exec_size: Tuple[int, int, int]
    args_dict: Dict[str, str]

    def __init__(
        self,
        shader_description: vc.ShaderDescription,
        func_args: List[Tuple[vc.BaseVariable, str, typing.Any]],
        my_local_size: Tuple[int, int, int],
        my_workgroups: Tuple[int, int, int],
        my_exec_size: Tuple[int, int, int],
        args_dict: Dict[str, str]
    ):
        self.shader_description = shader_description
        self.plan = vd.ComputePlan(shader_description.source, shader_description.binding_type_list, shader_description.pc_size, shader_description.name)
        self.func_args = func_args
        self.my_local_size = my_local_size
        self.my_workgroups = my_workgroups
        self.my_exec_size = my_exec_size
        self.args_dict = args_dict

    def __repr__(self) -> str:
        result = ""

        for ii, line in enumerate(self.shader_description.source.split("\n")):
            result += f"{ii + 1:4d}: {line}\n"

        return result

    def __call__(self, *args, **kwargs):
        my_blocks = (0, 0, 0)
        my_limits = (0, 0, 0)
        my_cmd_stream: vd.CommandStream = None

        if "workgroups" in kwargs or self.my_workgroups is not None:
            true_dims = sanitize_dims_tuple(
                self.func_args,
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
            true_dims = sanitize_dims_tuple(
                self.func_args,
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
        
        if "cmd_stream" in kwargs:
            if not isinstance(kwargs["cmd_stream"], vd.CommandStream):
                raise ValueError("Expected a CommandStream object for 'cmd_stream'!")

            my_cmd_stream = kwargs["cmd_stream"]
        
        if my_cmd_stream is None:
            my_cmd_stream = vd.global_cmd_stream()
        
        bound_buffers = []
        bound_images = []
        uniform_values = {}
        pc_values = {}

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
                
                bound_buffers.append((arg, arg_var[0].binding, arg_var[0].shape_name))

            elif isinstance(arg_var[0], vc.ImageVariable):
                if not isinstance(arg, vd.Image):
                    raise ValueError(f"Expected an image for argument '{arg_var[1]}'!")
                
                bound_images.append((arg, arg_var[0].binding))
            
            elif isinstance(arg_var[0], vc.ShaderVariable):
                if not arg_var[0]._varying:
                    if isinstance(arg, LaunchBindObject):
                        raise ValueError("Cannot use LaunchVariables for Constants")

                    uniform_values[arg_var[0].raw_name] = arg
                else:
                    if len(self.shader_description.pc_structure) == 0:
                        raise ValueError("Something went wrong with push constants!!")

                    if isinstance(arg, LaunchBindObject):
                        if my_cmd_stream.submit_on_record:
                            raise ValueError("Cannot bind Variables for default cmd list!")

                        arg.register_pc_var(pc_buff, arg_var[0].raw_name)
                    else:
                        pc_values[arg_var[0].raw_name] = arg
            else:
                raise ValueError(f"Something very wrong happened!")
        
        my_cmd_stream.record_shader(
            self.plan, 
            self.shader_description, 
            my_limits, 
            my_blocks, 
            bound_buffers, 
            bound_images, 
            uniform_values, 
            pc_values
        )
