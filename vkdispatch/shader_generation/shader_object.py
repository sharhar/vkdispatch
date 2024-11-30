import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Tuple
from typing import Union
from typing import Callable
from typing import List
from typing import Any
from typing import Dict
from typing import Optional

import uuid

import dataclasses

import numpy as np

def bounds_to_tuple(bounds: Union[int, Tuple[int, int, int], None]) -> Optional[Tuple[int, int, int]]:
    if bounds is None:
        return None

    if isinstance(bounds, int) or np.issubdtype(type(bounds), np.integer):
        return (bounds, 1, 1)

    if not isinstance(bounds, tuple):
        raise ValueError("Must provide a tuple of dimensions!")
    
    if len(bounds) < 1 or len(bounds) > 3:
        raise ValueError("Must provide a tuple of length 1, 2, or 3!")
    
    return_val = [1, 1, 1]

    for ii, val in enumerate(bounds):
        if not isinstance(val, int) and not np.issubdtype(type(val), np.integer):
            raise ValueError("All dimensions must be integers!")
        
        return_val[ii] = val
    
    return (return_val[0], return_val[1], return_val[2])

def params_holder_to_tuple(value, names_and_defaults, args, kwargs) -> Optional[Tuple[int, int, int]]:
    if callable(value):
        value = value(LaunchParametersHolder(names_and_defaults, args, kwargs))

    return bounds_to_tuple(value)

class LaunchParametersHolder:
    def __init__(self, names_and_defaults, args, kwargs) -> None:
        self.ref_dict = {}

        for ii, arg_var in enumerate(names_and_defaults):
            arg = None
            
            if ii < len(args):
                arg = args[ii]
            else:
                if arg_var[0] not in kwargs:
                    if arg_var[1] is None:
                        raise ValueError(f"Missing argument '{arg_var[0]}'!")
                    
                    arg = arg_var[1]
                else:
                    arg = kwargs[arg_var[0]]
            
            self.ref_dict[arg_var[0]] = arg

    def __getattr__(self, name: str):
        return self.ref_dict[name]

class ExectionBounds:
    local_size: Tuple[int, int, int]
    workgroups: Union[Tuple[int, int, int], Callable, None]
    exec_size: Union[Tuple[int, int, int], Callable, None]
    names_and_defaults: List[Tuple[str, Any]]

    def __init__(self, names_and_defaults, local_size, workgroups, exec_size) -> None:
        self.names_and_defaults = names_and_defaults
        self.local_size = local_size
        self.workgroups = workgroups
        self.exec_size = exec_size

    def get_blocks_and_limits(self, args, kwargs: dict) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        my_blocks = params_holder_to_tuple(kwargs.get("workgroups", self.workgroups), self.names_and_defaults, args, kwargs)
        my_limits = params_holder_to_tuple(kwargs.get("exec_size", self.exec_size) , self.names_and_defaults, args, kwargs)
        
        if my_limits is None:
            if my_blocks is None:
                raise ValueError("Must provide either 'exec_size' or 'workgroups'!")

            my_limits = (my_blocks[0] * self.local_size[0],
                            my_blocks[1] * self.local_size[1],
                            my_blocks[2] * self.local_size[2])
        else:
            my_blocks = ((my_limits[0] + self.local_size[0] - 1) // self.local_size[0],
                            (my_limits[1] + self.local_size[1] - 1) // self.local_size[1],
                            (my_limits[2] + self.local_size[2] - 1) // self.local_size[2])
        
        return (my_blocks, my_limits)

@dataclasses.dataclass
class CompiledShader:
    plan: vd.ComputePlan
    shader_description: vc.ShaderDescription
    bounds: ExectionBounds

execution_bound = Union[Tuple[int, int, int], Callable, None]

class ShaderObject:
    name: str
    builder: vc.ShaderBuilder
    shader_signature: vd.ShaderSignature
    shaders: Dict[Tuple[int, int, int], CompiledShader]

    default_local_size: Union[int, Tuple[int, int, int], Callable[[LaunchParametersHolder], Tuple[int, int, int]], None]
    default_workgroups: Union[int, Tuple[int, int, int], Callable[[LaunchParametersHolder], Tuple[int, int, int]], None]
    default_exec_size: Union[int, Tuple[int, int, int], Callable[[LaunchParametersHolder], Tuple[int, int, int]], None]

    def __init__(self, 
                 name: str, 
                 local_size: Union[int, Tuple[int, int, int], Callable[[LaunchParametersHolder], Tuple[int, int, int]], None] = None,
                 workgroups: Union[int, Tuple[int, int, int], Callable[[LaunchParametersHolder], Tuple[int, int, int]], None] = None, 
                 exec_size: Union[int, Tuple[int, int, int], Callable[[LaunchParametersHolder], Tuple[int, int, int]], None] = None) -> None:
        self.name = name 
        self.builder = vc.ShaderBuilder()
        self.shader_signature = vd.ShaderSignature()
        self.shaders = {}
        self.default_local_size = local_size
        self.default_exec_size = workgroups
        self.default_workgroups = exec_size
    
    def args_from_inspectable_function(self, func: Callable) -> List[vc.BaseVariable]:
        return self.shader_signature.make_from_inspectable_function(func, builder=self.builder)

    def args_from_type_annotations(self, signature: Tuple) -> List[vc.BaseVariable]:
        return self.shader_signature.make_from_type_annotations(signature, builder=self.builder)

    def build(self, local_size: Union[Tuple[int, int, int], None] = None):
        true_local_size = (
            local_size
            if local_size is not None
            else (vd.get_context().max_workgroup_size[0], 1, 1)
        )

        bounds = ExectionBounds(self.shader_signature.get_names_and_defaults(), true_local_size, self.default_workgroups, self.default_exec_size)

        shader_description = self.builder.build(
            true_local_size[0], true_local_size[1], true_local_size[2], self.name
        )

        plan = vd.ComputePlan(
            shader_description.source, 
            shader_description.binding_type_list, 
            shader_description.pc_size, 
            shader_description.name
        )

        self.shaders[local_size] = CompiledShader(plan, shader_description, bounds)

    def __repr__(self) -> str:
        if len(self.shaders) == 0:
            shader_description = self.builder.build(0, 0, 0, self.name)

            for ii, line in enumerate(shader_description.source.split("\n")):
                result += f"{ii + 1:4d}: {line}\n"

            return result

        result = ""

        for key, value in self.shaders.items():
            result += f"Local Size: {key}\n"
            result += f"  {value}\n"

            for ii, line in enumerate(value.shader_description.source.split("\n")):
                result += f"{ii + 1:4d}: {line}\n"

        return result

    def __call__(self, *args, **kwargs):
        local_size = params_holder_to_tuple(
            kwargs.get("local_size", self.default_local_size),
            self.shader_signature.get_names_and_defaults(),
            args, kwargs
        )

        if local_size not in self.shaders.keys():
            self.build(local_size)

        my_blocks, my_limits = self.shaders[local_size].bounds.get_blocks_and_limits(args, kwargs)

        print(f"Local Size: {local_size}")
        print(f"Blocks: {my_blocks}, Limits: {my_limits}")


        my_cmd_stream: vd.CommandStream = None

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

        shader_uuid = f"{self.shaders[local_size].shader_description.name}.{uuid.uuid4()}"

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
                if len(self.shaders[local_size].shader_description.pc_structure) == 0:
                    raise ValueError("Something went wrong with push constants!!")

                if callable(arg): # isinstance(arg, LaunchBindObject):
                    if my_cmd_stream.submit_on_record:
                        raise ValueError("Cannot bind Variables for default cmd list!")
                    
                    arg((shader_uuid, shader_arg.shader_name))
                else:
                    pc_values[shader_arg.shader_name] = arg
            else:
                raise ValueError(f"Something very wrong happened!")
        
        my_cmd_stream.record_shader(
            self.shaders[local_size].plan, 
            self.shaders[local_size].shader_description, 
            my_limits, 
            my_blocks, 
            bound_buffers, 
            bound_images, 
            uniform_values, 
            pc_values,
            shader_uuid=shader_uuid
        )

