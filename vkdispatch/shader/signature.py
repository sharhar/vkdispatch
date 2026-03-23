import vkdispatch.codegen as vc

from ..base.dtype import is_dtype

from typing import List
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Dict

from typing import get_type_hints

import dataclasses
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

    def __init__(self, 
                 name: str,
                 arg_type: ShaderArgumentType,
                 default_value: Any,
                 shader_name: Union[str, Dict[str, str]],
                 shader_shape_name: Optional[str] = None,
                 binding: Optional[int] = None):
        self.name = name
        self.arg_type = arg_type
        self.default_value = default_value
        self.shader_name = shader_name
        self.shader_shape_name = shader_shape_name
        self.binding = binding

def annotation_to_shader_arg_and_variable(builder: vc.ShaderBuilder, type_annotation: Any, name: str, default_value: Any):
    # Dataclass case
    if(dataclasses.is_dataclass(type_annotation)):
        creation_args: Dict[str, vc.ShaderVariable] = {}
        value_name = {}

        for field_name, field_type in get_type_hints(type_annotation).items():
            assert is_dtype(field_type), f"Unsupported type '{field_type}' for field '{type_annotation}.{field_name}'"

            creation_args[field_name] = builder.declare_constant(field_type)
            value_name[field_name] = creation_args[field_name].raw_name

        return ShaderArgument(
            name,
            ShaderArgumentType.CONSTANT_DATACLASS,
            default_value,
            value_name
        ), type_annotation(**creation_args)
    
    # Buffer case
    if(issubclass(type_annotation.__origin__, vc.Buffer)):
        shader_var = builder.declare_buffer(type_annotation.__args__[0])

        return ShaderArgument(
            name,
            ShaderArgumentType.BUFFER,
            default_value,
            shader_var.raw_name,
            shader_shape_name=shader_var.shape_name,
            binding=shader_var.binding
        ), shader_var

    # Image case
    if(issubclass(type_annotation.__origin__, vc.ImageVariable)):
        shader_var = builder.declare_image(
            type_annotation.__origin__.dimensions
        )

        return ShaderArgument(
            name,
            ShaderArgumentType.IMAGE,
            default_value,
            shader_var.raw_name,
            binding=shader_var.binding
        ), shader_var

    # if(issubclass(type_annotation.__origin__, vc.Image2D)):
    #     shader_param = builder.declare_image(2)
    #     arg_type = ShaderArgumentType.IMAGE
    #     binding = shader_param.binding
    #     value_name = shader_param.raw_name

    # if(issubclass(type_annotation.__origin__, vc.Image3D)):
    #     shader_param = builder.declare_image(3)
    #     arg_type = ShaderArgumentType.IMAGE
    #     binding = shader_param.binding
    #     value_name = shader_param.raw_name

    if(issubclass(type_annotation.__origin__, vc.Constant)):
        shader_var = builder.declare_constant(type_annotation.__args__[0])

        return ShaderArgument(
            name,
            ShaderArgumentType.CONSTANT,
            default_value,
            shader_var.raw_name
        ), shader_var

    if(issubclass(type_annotation.__origin__, vc.Variable)):
        shader_var = builder.declare_variable(type_annotation.__args__[0])

        return ShaderArgument(
            name,
            ShaderArgumentType.VARIABLE,
            default_value,
            shader_var.raw_name
        ), shader_var

    raise ValueError(f"Unsupported type '{type_annotation.__args__[0]}'")

class ShaderSignature:
    arguments: List[ShaderArgument]
    variables: List[vc.ShaderVariable]

    def __init__(self,
                 builder: vc.ShaderBuilder,
                 type_annotations: List,
                 names: Optional[List[str]] = None,
                 defaults: Optional[List[Any]] = None) -> 'ShaderSignature':
        self.arguments = []
        self.variables = []

        for i in range(len(type_annotations)):
            shader_arg, shader_var = annotation_to_shader_arg_and_variable(
                builder,
                type_annotations[i],
                names[i] if names is not None else f"param{i}",
                defaults[i] if defaults is not None else None
            )

            self.variables.append(shader_var)
            self.arguments.append(shader_arg)
    
    def get_variables(self) -> List[vc.ShaderVariable]:
        return self.variables

    def get_names_and_defaults(self) -> List[Tuple[str, Any]]:
        return [(arg.name, arg.default_value) for arg in self.arguments]
