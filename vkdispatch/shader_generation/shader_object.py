import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Tuple
from typing import Union
from typing import Callable
from typing import List
from typing import Any

import uuid

import dataclasses

import numpy as np

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

    def process_input(self, in_val, args, kwargs) -> Tuple[int, int, int]:
        if callable(in_val):
            in_val = in_val(LaunchParametersHolder(self.names_and_defaults, args, kwargs))

        if isinstance(in_val, int) or np.issubdtype(type(in_val), np.integer):
            return (in_val, 1, 1) # type: ignore

        if not isinstance(in_val, tuple):
            raise ValueError("Must provide a tuple of dimensions!")
        
        if len(in_val) < 0 or len(in_val) > 4:
            raise ValueError("Must provide a tuple of length 1, 2, or 3!")
        
        return_val = [1, 1, 1]

        for ii, val in enumerate(in_val):
            if not isinstance(val, int) and not np.issubdtype(type(val), np.integer):
                raise ValueError("All dimensions must be integers!")
            
            return_val[ii] = val
        
        return (return_val[0], return_val[1], return_val[2])

    def get_blocks_and_limits(self, args, kwargs) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        my_blocks = (0, 0, 0)
        my_limits = (0, 0, 0)
        
        if "workgroups" in kwargs or self.workgroups is not None:
            true_dims = self.process_input(
                kwargs["workgroups"]
                if "workgroups" in kwargs
                else self.workgroups,
                args, kwargs
            )

            my_blocks = true_dims
            my_limits = (true_dims[0] * self.local_size[0],
                            true_dims[1] * self.local_size[1],
                            true_dims[2] * self.local_size[2])
        
        if "exec_size" in kwargs or self.exec_size is not None:
            true_dims = self.process_input(
                kwargs["exec_size"]
                if "exec_size" in kwargs
                else self.exec_size,
                args, kwargs
            )

            my_limits = true_dims
            my_blocks = ((true_dims[0] + self.local_size[0] - 1) // self.local_size[0],
                            (true_dims[1] + self.local_size[1] - 1) // self.local_size[1],
                            (true_dims[2] + self.local_size[2] - 1) // self.local_size[2])

        if my_blocks is None:
            raise ValueError("Must provide either 'exec_size' or 'workgroups'!")
        
        return (my_blocks, my_limits)

class ShaderObject:
    name: str
    plan: vd.ComputePlan
    shader_description: vc.ShaderDescription
    shader_signature: vd.ShaderSignature
    bounds: ExectionBounds
    ready: bool
    source: str

    def __init__(self, name: str, description: vc.ShaderDescription, signature: vd.ShaderSignature) -> None:
        self.name = name 
        self.plan = None
        self.shader_description = description
        self.shader_signature = signature
        self.bounds = None
        self.ready = False
        self.source = None

    def build(self, local_size: Tuple[int, int, int] = None, workgroups=None, exec_size=None):
        assert not self.ready, "Cannot build a shader that is already built!"

        my_local_size = (
            local_size
            if local_size is not None
            else [vd.get_context().max_workgroup_size[0], 1, 1]
        )

        self.bounds = ExectionBounds(self.shader_signature.get_names_and_defaults(), my_local_size, workgroups, exec_size)

        self.source = vc.get_source_from_description(
            self.shader_description, my_local_size[0], my_local_size[1], my_local_size[2] #, self.name
        )

        self.plan = vd.ComputePlan(
            self.source, 
            self.shader_description.binding_type_list, 
            self.shader_description.pc_size, 
            self.shader_description.name
        )

        self.ready = True

    def __repr__(self) -> str:
        result = ""

        for ii, line in enumerate(self.source.split("\n")):
            result += f"{ii + 1:4d}: {line}\n"

        return result

    def __call__(self, *args, **kwargs):
        vd.make_context()

        if not self.ready:
            raise ValueError("Cannot call a shader that is not built!")

        my_blocks, my_limits = self.bounds.get_blocks_and_limits(args, kwargs)

        my_cmd_stream: vd.CommandStream = None

        if "cmd_stream" in kwargs and kwargs["cmd_stream"] is not None:
            if not isinstance(kwargs["cmd_stream"], vd.CommandStream):
                raise ValueError("Expected a CommandStream object for 'cmd_stream'!")

            my_cmd_stream = kwargs["cmd_stream"]
        
        if my_cmd_stream is None:
            my_cmd_stream = vd.global_cmd_stream()
        
        bound_buffers = []
        bound_samplers = []
        uniform_values = {}
        pc_values = {}

        shader_uuid = f"{self.shader_description.name}.{uuid.uuid4()}"

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
                if not isinstance(arg, vd.Sampler):
                    raise ValueError(f"Expected an image for argument '{shader_arg.name}'!")
                
                bound_samplers.append((arg, shader_arg.binding))
            
            elif shader_arg.arg_type == vd.ShaderArgumentType.CONSTANT:
                if callable(arg): # isinstance(arg, LaunchBindObject):
                    raise ValueError("Cannot use LaunchVariables for Constants")

                uniform_values[shader_arg.shader_name] = arg
            
            elif shader_arg.arg_type == vd.ShaderArgumentType.CONSTANT_DATACLASS:
                if callable(arg):
                    raise ValueError("Cannot use LaunchVariables for Constants")
                
                for field in dataclasses.fields(arg):
                    uniform_values[shader_arg.shader_name[field.name]] = getattr(arg, field.name)

            elif shader_arg.arg_type == vd.ShaderArgumentType.VARIABLE:
                if len(self.shader_description.pc_structure) == 0:
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
            self.plan, 
            self.shader_description, 
            my_limits, 
            my_blocks, 
            bound_buffers, 
            bound_samplers, 
            uniform_values, 
            pc_values,
            shader_uuid=shader_uuid
        )

