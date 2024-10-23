import vkdispatch.codegen as vc

from typing import List
from typing import Any
from typing import Callable
from typing import Optional
#from types import GenericAlias

import dataclasses

import inspect

import enum

class ShaderArgumentType(enum.Enum):
    BUFFER = 0
    IMAGE = 1
    VARIABLE = 2
    CONSTANT = 3

@dataclasses.dataclass
class ShaderArgument:
    name: str
    arg_type: ShaderArgumentType
    default_value: Any
    shader_name: str
    shader_shape_name: Optional[str]
    binding: Optional[int]


class ShaderSignature:
    arguments: List[ShaderArgument]

    def __init__(self, arguments: Optional[List[ShaderArgument]] = None) -> None:
        self.arguments = arguments if arguments is not None else []

    def make_from_inspectable_function(self, func: Callable, builder: Optional[vc.ShaderBuilder] = None) -> List[vc.BaseVariable]:
        func_signature = inspect.signature(func)

        annotations = []
        names = []
        defaults = []

        for param in func_signature.parameters.values():
            if param.annotation == inspect.Parameter.empty:
                raise ValueError("All parameters must be annotated")

            if not hasattr(param.annotation, '__args__'):
                raise TypeError(f"Argument '{param.name}: vd.{param.annotation}' must have a type annotation")
            
            if len(param.annotation.__args__) != 1:
                raise ValueError(f"Type '{param.name}: vd.{param.annotation.__name__}' must have exactly one type argument")

            annotations.append(param.annotation)
            names.append(param.name)
            defaults.append(param.default if param.default != inspect.Parameter.empty else None)
        
        return self.make_from_type_annotations(annotations, names, defaults, builder)

    def make_from_type_annotations(self, 
                       annotations: List, # [GenericAlias], adding this type annotation causes an error in python 3.8, so for now it is left as List
                       names: Optional[List[str]] = None,
                       defaults: Optional[List[Any]] = None,
                       builder: Optional[vc.ShaderBuilder] = None) -> List[vc.BaseVariable]:
        
        if builder is None:
            builder = vc.builder_obj

        shader_function_paramaters: List[vc.BaseVariable] = []

        for i in range(len(annotations)):
            type_arg = annotations[i].__args__[0]

            arg_type = None
            shape_name = None
            binding = None

            if(issubclass(annotations[i].__origin__, vc.Buffer)):
                shader_param = builder.declare_buffer(type_arg)

                arg_type = ShaderArgumentType.BUFFER
                shape_name = shader_param.shape_name
                binding = shader_param.binding

                shader_function_paramaters.append(shader_param)

            elif(issubclass(annotations[i].__origin__, vc.Image1D)):
                shader_param = builder.declare_image(1)
                
                arg_type = ShaderArgumentType.IMAGE
                binding = shader_param.binding

                shader_function_paramaters.append(shader_param)

            elif(issubclass(annotations[i].__origin__, vc.Image2D)):
                shader_param = builder.declare_image(2)
                
                arg_type = ShaderArgumentType.IMAGE
                binding = shader_param.binding

                shader_function_paramaters.append(shader_param)
                
            elif(issubclass(annotations[i].__origin__, vc.Image3D)):
                shader_param = builder.declare_image(3)
                
                arg_type = ShaderArgumentType.IMAGE
                binding = shader_param.binding

                shader_function_paramaters.append(shader_param)
                
            elif(issubclass(annotations[i].__origin__, vc.Constant)):
                shader_function_paramaters.append(builder.declare_constant(type_arg))
                arg_type = ShaderArgumentType.CONSTANT

            elif(issubclass(annotations[i].__origin__, vc.Variable)):
                shader_function_paramaters.append(builder.declare_variable(type_arg))
                arg_type = ShaderArgumentType.VARIABLE

            else:
                raise ValueError(f"Unsupported type '{type_arg}'")

            self.arguments.append(ShaderArgument(
                names[i] if names is not None else f"param{i}",
                arg_type,
                defaults[i] if defaults is not None else None,
                shader_function_paramaters[-1].raw_name,
                shape_name,
                binding
            ))
    
        return shader_function_paramaters
    
    def get_func_args(self) -> List[vc.BaseVariable]:
        return [(arg.shader_name, arg.name, arg.default_value) for arg in self.arguments]
