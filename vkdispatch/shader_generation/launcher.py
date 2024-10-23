import vkdispatch as vd
import vkdispatch.codegen as vc

import dataclasses

import copy

import typing

from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from typing import Callable

import numpy as np

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

# class LaunchVariables:
#     def __init__(self, cmd_stream: vd.CommandStream) -> None:
#         self.name_to_key_dict = {}
#         self.cmd_stream = cmd_stream

#         self.queued_ops = {}

#     def new(self, name: str):
#         if name in self.name_to_key_dict.keys():
#             raise ValueError("Variable already bound!")
        
#         def register_var(key: Tuple[str, str]):
#             self.name_to_key_dict[name] = key

#         return register_var

#     def set(self, name: str, value: typing.Any):
#         if name not in self.name_to_key_dict.keys():
#             raise ValueError("Variable not bound!")
        
#         self.queued_ops[self.name_to_key_dict[name]] = value

#     def build(self):
#         for key, val in self.queued_ops.items():
#             self.cmd_stream.pc_builder[key] = val

class ShaderLauncher:
    plan: vd.ComputePlan
    shader_description: vc.ShaderDescription
    shader_signature: vd.ShaderSignature
    my_local_size: Tuple[int, int, int]
    my_workgroups: Tuple[int, int, int]
    my_exec_size: Tuple[int, int, int]

    def __init__(
        self,
        description: vc.ShaderDescription,
        signature: vd.ShaderSignature,
        my_local_size: Tuple[int, int, int],
        my_workgroups: Tuple[int, int, int],
        my_exec_size: Tuple[int, int, int],
    ):
        self.shader_description = description
        self.plan = vd.ComputePlan(description.source, description.binding_type_list, description.pc_size, description.name)
        self.shader_signature = signature
        self.my_local_size = my_local_size
        self.my_workgroups = my_workgroups
        self.my_exec_size = my_exec_size

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
                self.shader_signature.get_func_args(),
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
                self.shader_signature.get_func_args(),
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

        for ii, shader_arg in enumerate(self.shader_signature.arguments):
            arg = None
            
            if ii < len(args):
                arg = args[ii]
            else:
                if shader_arg.name not in kwargs:
                    if shader_arg.default_value is None:
                        raise ValueError(f"Missing argument '{shader_arg.name}'!")
                    
                    arg = shader_arg.default_value
                else:
                    arg = kwargs[shader_arg.name]

            if shader_arg.arg_type == vd.ShaderArgumentType.BUFFER:
                if not isinstance(arg, vd.Buffer):
                    raise ValueError(f"Expected a buffer for argument '{shader_arg.name}'!")
                
                bound_buffers.append((arg, shader_arg.binding, shader_arg.shader_shape_name))

            elif shader_arg.arg_type == vd.ShaderArgumentType.IMAGE:
                if not isinstance(arg, vd.Image):
                    raise ValueError(f"Expected an image for argument '{shader_arg.name}'!")
                
                bound_images.append((arg, shader_arg.binding))
            
            elif shader_arg.arg_type == vd.ShaderArgumentType.CONSTANT:
                if callable(arg): # isinstance(arg, LaunchBindObject):
                    raise ValueError("Cannot use LaunchVariables for Constants")

                uniform_values[shader_arg.shader_name] = arg

            elif shader_arg.arg_type == vd.ShaderArgumentType.VARIABLE:
                if len(self.shader_description.pc_structure) == 0:
                    raise ValueError("Something went wrong with push constants!!")

                if callable(arg): # isinstance(arg, LaunchBindObject):
                    if my_cmd_stream.submit_on_record:
                        raise ValueError("Cannot bind Variables for default cmd list!")
                    
                    arg((self.shader_description.name, shader_arg.shader_name))
                else:
                    pc_values[shader_arg.shader_name] = arg
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
