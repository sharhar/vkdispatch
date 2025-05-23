import vkdispatch.codegen as vc

from ..base.dtype import is_dtype

from typing import List
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Dict
#from types import GenericAlias

from typing import get_type_hints

import dataclasses

import inspect

import enum

class ShaderArgumentType(enum.Enum):
    BUFFER = 0
    IMAGE = 1
    VARIABLE = 2
    CONSTANT = 3
    CONSTANT_DATACLASS = 4

@dataclasses.dataclass
class ShaderArgument:
    name: str
    arg_type: ShaderArgumentType
    default_value: Any
    shader_name: Union[str, Dict[str, str]]
    shader_shape_name: Optional[str]
    binding: Optional[int]


class ShaderSignature:
    arguments: List[ShaderArgument]
    variables: List[vc.BaseVariable]

    def __init__(self):
        raise NotImplementedError("This class is not meant to be instantiated")

    @classmethod
    def from_argument_list(cls, arguments: Optional[List[ShaderArgument]] = None) -> 'ShaderSignature':
        instance = cls.__new__(cls)  # Bypasses __init__
        instance.arguments = arguments if arguments is not None else []
        instance.variables = None
        return instance

    @classmethod
    def from_inspectable_function(cls, builder: vc.ShaderBuilder, func: Callable) -> 'ShaderSignature':
        func_signature = inspect.signature(func)

        annotations = []
        names = []
        defaults = []

        for param in func_signature.parameters.values():
            if param.annotation == inspect.Parameter.empty:
                raise ValueError("All parameters must be annotated")


            if not dataclasses.is_dataclass(param.annotation): # issubclass(param.annotation.__origin__, dataclasses.dataclass):
                if not hasattr(param.annotation, '__args__'):
                    raise TypeError(f"Argument '{param.name}: vd.{param.annotation}' must have a type annotation")
                
                if len(param.annotation.__args__) != 1:
                    raise ValueError(f"Type '{param.name}: vd.{param.annotation.__name__}' must have exactly one type argument")

            annotations.append(param.annotation)
            names.append(param.name)
            defaults.append(param.default if param.default != inspect.Parameter.empty else None)
        
        return ShaderSignature.from_type_annotations(builder, annotations, names, defaults)

    @classmethod
    def from_type_annotations(cls, 
                       builder: vc.ShaderBuilder,
                       annotations: List,
                       names: Optional[List[str]] = None,
                       defaults: Optional[List[Any]] = None) -> 'ShaderSignature':

        instance = cls.__new__(cls)  # Bypasses __init__
        instance.arguments = []
        instance.variables = []

        for i in range(len(annotations)):
            shader_param = None
            arg_type = None
            shape_name = None
            binding = None
            value_name = None

            if(dataclasses.is_dataclass(annotations[i])):
                creation_args: Dict[str, vc.ShaderVariable] = {}
                arg_type = ShaderArgumentType.CONSTANT_DATACLASS
                value_name = {}

                for field_name, field_type in get_type_hints(annotations[i]).items():
                    assert is_dtype(field_type), f"Unsupported type '{field_type}' for field '{annotations[i]}.{field_name}'"

                    creation_args[field_name] = builder.declare_constant(field_type)
                    value_name[field_name] = creation_args[field_name].raw_name

                shader_param = annotations[i](**creation_args)
            
            elif(issubclass(annotations[i].__origin__, vc.Buffer)):
                shader_param = builder.declare_buffer(annotations[i].__args__[0])

                arg_type = ShaderArgumentType.BUFFER
                shape_name = shader_param.shape_name
                binding = shader_param.binding
                value_name = shader_param.raw_name

            elif(issubclass(annotations[i].__origin__, vc.Image1D)):
                shader_param = builder.declare_image(1)
                
                arg_type = ShaderArgumentType.IMAGE
                binding = shader_param.binding
                value_name = shader_param.raw_name

            elif(issubclass(annotations[i].__origin__, vc.Image2D)):
                shader_param = builder.declare_image(2)
                arg_type = ShaderArgumentType.IMAGE
                binding = shader_param.binding
                value_name = shader_param.raw_name

            elif(issubclass(annotations[i].__origin__, vc.Image3D)):
                shader_param = builder.declare_image(3)
                arg_type = ShaderArgumentType.IMAGE
                binding = shader_param.binding
                value_name = shader_param.raw_name

            elif(issubclass(annotations[i].__origin__, vc.Constant)):
                shader_param = builder.declare_constant(annotations[i].__args__[0])
                value_name = shader_param.raw_name
                arg_type = ShaderArgumentType.CONSTANT
            elif(issubclass(annotations[i].__origin__, vc.Variable)):
                shader_param = builder.declare_variable(annotations[i].__args__[0])
                arg_type = ShaderArgumentType.VARIABLE
                value_name = shader_param.raw_name

            else:
                raise ValueError(f"Unsupported type '{annotations[i].__args__[0]}'")

            instance.variables.append(shader_param)
            
            instance.arguments.append(ShaderArgument(
                names[i] if names is not None else f"param{i}",
                arg_type,
                defaults[i] if defaults is not None else None,
                value_name,
                shape_name,
                binding
            ))
    
        return instance
    
    def get_variables(self) -> List[vc.BaseVariable]:
        return self.variables

    def get_names_and_defaults(self) -> List[Tuple[str, Any]]:
        return [(arg.name, arg.default_value) for arg in self.arguments]
    
#    def get_func_args(self) -> List[Tuple[str, str, Any]]:
#        return [(arg.shader_name, arg.name, arg.default_value) for arg in self.arguments]
