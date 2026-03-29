import enum
import dataclasses
from typing import List, Tuple, Optional, Any, Union, Dict

from .backends import CodeGenBackend
from .struct_builder import StructElement

class BindingType(enum.Enum):
    """
    A dataclass that represents the type of a binding in a shader. Either a
    STORAGE_BUFFER, UNIFORM_BUFFER, or SAMPLER.
    """
    STORAGE_BUFFER = 1
    UNIFORM_BUFFER = 3
    SAMPLER = 5

class ShaderArgumentType(enum.Enum):
    BUFFER = 0
    IMAGE = 1
    VARIABLE = 2
    CONSTANT = 3
    CONSTANT_DATACLASS = 4

@dataclasses.dataclass
class ShaderArgumentInfo:
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

@dataclasses.dataclass
class ShaderDescription:
    """
    A dataclass that represents a description of a shader object.

    Attributes:
        source (str): The source code of the shader.
        pc_size (int): The size of the push constant buffer in bytes.
        pc_structure (List[vc.StructElement]): The structure of the push constant buffer.
        uniform_structure (List[vc.StructElement]): The structure of the uniform buffer.
        binding_type_list (List[BindingType]): The list of binding types.
    """

    header: str
    body: str
    name: str
    pc_size: int
    pc_structure: List[StructElement]
    uniform_structure: List[StructElement]
    binding_type_list: List[BindingType]
    binding_access: List[Tuple[bool, bool]] # List of tuples indicating read and write access for each binding
    exec_count_name: Optional[str]
    resource_binding_base: int
    backend: CodeGenBackend
    shader_arg_infos: List[ShaderArgumentInfo]

    def make_source(self, x: int, y: int, z: int) -> str:
        return self.backend.make_source(self.header, self.body, x, y, z)
    
    def get_arg_names_and_defaults(self) -> List[Tuple[str, Any]]:
        return [(arg.name, arg.default_value) for arg in self.shader_arg_infos]
    
    def __repr__(self):
        description_string = ""

        description_string += f"Shader Name: {self.name}\n"
        description_string += f"Push Constant Size: {self.pc_size} bytes\n"
        description_string += f"Push Constant Structure: {self.pc_structure}\n"
        description_string += f"Uniform Structure: {self.uniform_structure}\n"
        description_string += f"Binding Types: {self.binding_type_list}\n"
        description_string += f"Binding Access: {self.binding_access}\n"
        description_string += f"Execution Count Name: {self.exec_count_name}\n"
        description_string += f"Backend: {self.backend.name if self.backend is not None else 'none'}\n"
        description_string += f"Header:\n{self.header}\n"
        description_string += f"Body:\n{self.body}\n"
        return description_string