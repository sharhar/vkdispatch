import vkdispatch.base.dtype as dtypes

from .struct_builder import StructElement, StructBuilder

from .shader_writer import ShaderWriter
from .backends import CodeGenBackend
from .global_builder import get_codegen_backend

from enum import IntFlag, auto

from typing import Dict, List, Optional, Tuple

import dataclasses

import enum

from .variables.variables import BaseVariable, ShaderVariable, ScaledAndOfftsetIntVariable
from .variables.bound_variables import BufferVariable, ImageVariable

@dataclasses.dataclass
class SharedBuffer:
    """
    A dataclass that represents a shared buffer in a shader.

    Attributes:
        dtype (vd.dtype): The dtype of the shared buffer.
        size (int): The size of the shared buffer.
        name (str): The name of the shared buffer within the shader code.
    """
    dtype: dtypes.dtype
    size: int
    name: str

class BindingType(enum.Enum):
    """
    A dataclass that represents the type of a binding in a shader. Either a
    STORAGE_BUFFER, UNIFORM_BUFFER, or SAMPLER.
    """
    STORAGE_BUFFER = 1
    UNIFORM_BUFFER = 3
    SAMPLER = 5

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
    backend: Optional[CodeGenBackend] = None

    def make_source(self, x: int, y: int, z: int) -> str:
        if self.backend is None:
            layout_str = f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;"
            return f"{self.header}\n{layout_str}\n{self.body}"

        return self.backend.make_source(self.header, self.body, x, y, z)
    
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

@dataclasses.dataclass
class ShaderBinding:
    """
    A dataclass that represents a bound resource in a shader. Either a 
    buffer or an image.

    Attributes:
        dtype (vd.dtype): The dtype of the resource. If 
            the resource is an image, this should be vd.vec4 
            (since all images are sampled with 4 channels in shaders).
        name (str): The name of the resource within the shader code.
        dimension (int): The dimension of the resource. Set to 0 for
            buffers and 1, 2, or 3 for images.
        binding_type (BindingType): The type of the binding. Either
            STORAGE_BUFFER, UNIFORM_BUFFER, or SAMPLER.
    """
    dtype: dtypes.dtype
    name: str
    dimension: int
    binding_type: BindingType

class ShaderFlags(IntFlag):
    NONE  = 0
    NO_SUBGROUP_OPS = auto()
    NO_PRINTF = auto()
    NO_EXEC_BOUNDS = auto()

class ShaderBuilder(ShaderWriter):
    binding_count: int
    binding_read_access: Dict[int, bool]
    binding_write_access: Dict[int, bool]
    binding_list: List[ShaderBinding]
    shared_buffers: List[SharedBuffer]
    scope_num: int
    pc_struct: StructBuilder
    uniform_struct: StructBuilder
    exec_count: Optional[ShaderVariable]
    flags: ShaderFlags
    backend: CodeGenBackend

    def __init__(self,
                 flags: ShaderFlags = ShaderFlags.NONE,
                 is_apple_device: bool = False,
                 backend: Optional[CodeGenBackend] = None) -> None:
        super().__init__()

        self.flags = flags
        self.is_apple_device = is_apple_device
        if backend is not None:
            self.backend = backend
        else:
            # Use the selected backend type while keeping per-builder backend state isolated.
            self.backend = get_codegen_backend().__class__()
        
        self.reset()

    def reset(self) -> None:
        self.backend.reset_state()
        self.binding_count = 0
        self.pc_struct = StructBuilder()
        self.uniform_struct = StructBuilder()
        self.binding_list = []
        self.binding_read_access = {}
        self.binding_write_access = {}
        self.shared_buffers = []
        self.scope_num = 1
        
        self.exec_count = None

        if not (self.flags & ShaderFlags.NO_EXEC_BOUNDS):
            self.exec_count = self.declare_constant(dtypes.uvec4, var_name="exec_count")
            self.append_contents(self.backend.exec_bounds_guard(self.exec_count.resolve()))

    def new_var(self,
                var_type: dtypes.dtype,
                name: str,
                parents: List["ShaderVariable"],
                lexical_unit: bool = False,
                settable: bool = False,
                register: bool = False) -> "ShaderVariable":
        return ShaderVariable(var_type,
                              name,
                              lexical_unit=lexical_unit,
                              settable=settable,
                              register=register,
                              parents=parents)
    
    def new_scaled_var(self,
                        var_type: dtypes.dtype,
                        name: str,
                        scale: int = 1,
                        offset: int = 0,
                        parents: List[BaseVariable] = None):
        return ScaledAndOfftsetIntVariable(var_type,
                                           name,
                                           scale=scale,
                                           offset=offset,
                                           parents=parents)

    def declare_constant(self, var_type: dtypes.dtype, count: int = 1, var_name: Optional[str] = None):
        if var_name is None:
            var_name = self.new_name()

        new_var = ShaderVariable(
            var_type=var_type,
            name=f"{self.backend.constant_namespace()}.{var_name}",
            raw_name=var_name,
            lexical_unit=True,
            settable=False,
            parents=[]
        )

        if count > 1:
            new_var.use_child_type = False
            new_var.can_index = True

        self.uniform_struct.register_element(new_var.raw_name, var_type, count)
        return new_var

    def declare_variable(self, var_type: dtypes.dtype, count: int = 1, var_name: Optional[str] = None):
        if self.backend.name == "cuda":
            raise NotImplementedError("Push Constants are not supported for the CUDA backend")

        if var_name is None:
            var_name = self.new_name()

        new_var = ShaderVariable(
            var_type=var_type,
            name=f"{self.backend.variable_namespace()}.{var_name}",
            raw_name=var_name,
            lexical_unit=True,
            settable=False,
            parents=[]
        )

        if count > 1:
            new_var.use_child_type = False
            new_var.can_index = True

        self.pc_struct.register_element(new_var.raw_name, var_type, count)
        return new_var
    
    def declare_buffer(self, var_type: dtypes.dtype, var_name: Optional[str] = None):
        self.binding_count += 1

        buffer_name = f"buf{self.binding_count}" if var_name is None else var_name
        shape_name = f"{buffer_name}_shape"
        
        self.binding_list.append(ShaderBinding(var_type, buffer_name, 0, BindingType.STORAGE_BUFFER))
        self.binding_read_access[self.binding_count] = False
        self.binding_write_access[self.binding_count] = False

        current_binding_count = self.binding_count

        def read_lambda():
            self.binding_read_access[current_binding_count] = True

        def write_lambda():
            self.binding_write_access[current_binding_count] = True

        def shape_var_factory():
            return self.declare_constant(dtypes.ivec4, var_name=shape_name)
        
        return BufferVariable(
            var_type,
            self.binding_count,
            f"{buffer_name}.data",
            shape_var_factory=shape_var_factory,
            shape_name=shape_name,
            read_lambda=read_lambda,
            write_lambda=write_lambda
        )
    
    def declare_image(self, dimensions: int, var_name: Optional[str] = None):
        self.binding_count += 1

        image_name = f"tex{self.binding_count}" if var_name is None else var_name
        self.binding_list.append(ShaderBinding(dtypes.vec4, image_name, dimensions, BindingType.SAMPLER))
        self.binding_read_access[self.binding_count] = False
        self.binding_write_access[self.binding_count] = False

        def read_lambda():
            self.binding_read_access[self.binding_count] = True

        def write_lambda():
            self.binding_write_access[self.binding_count] = True
        
        return ImageVariable(
            dtypes.vec4,
            self.binding_count,
            dimensions,
            f"{image_name}",
            read_lambda=read_lambda,
            write_lambda=write_lambda
        )
    
    def shared_buffer(self, var_type: dtypes.dtype, size: int, var_name: Optional[str] = None):
        if var_name is None:
            var_name = self.new_name()
        
        shape_name = f"{var_name}_shape"

        def shape_var_factory():
            return self.declare_constant(dtypes.ivec4, var_name=shape_name)

        new_var = BufferVariable(
            var_type,
            -1,
            var_name,
            shape_var_factory=shape_var_factory,
            shape_name=shape_name,
            read_lambda=lambda: None,
            write_lambda=lambda: None
        )

        self.shared_buffers.append(SharedBuffer(var_type, size, new_var.name))

        return new_var
    
    def compose_struct_decleration(self, elements: List[StructElement]) -> str:
        declerations = []

        for elem in elements:
            decleration_type = self.backend.type_name(elem.dtype)

            decleration_suffix = ""
            if elem.count > 1:
                decleration_suffix = f"[{elem.count}]"

            declerations.append(f"    {decleration_type} {elem.name}{decleration_suffix};")
        
        return "\n".join(declerations)

    def build(self, name: str) -> ShaderDescription:
        header = ""

        for shared_buffer in self.shared_buffers:
            header += self.backend.shared_buffer_declaration(
                shared_buffer.dtype,
                shared_buffer.name,
                shared_buffer.size
            ) + "\n"

        uniform_elements = self.uniform_struct.build()
        
        uniform_decleration_contents = self.compose_struct_decleration(uniform_elements)
        has_uniform_buffer = len(uniform_decleration_contents) > 0
        if has_uniform_buffer:
            header += self.backend.uniform_block_declaration(uniform_decleration_contents)

        binding_base = 1 if has_uniform_buffer else 0
        binding_type_list = []
        binding_access = []
        if has_uniform_buffer:
            binding_type_list.append(BindingType.UNIFORM_BUFFER)
            binding_access.append((True, False))  # UBO is read-only
        
        for ii, binding in enumerate(self.binding_list):
            emitted_binding = ii + binding_base
            if binding.binding_type == BindingType.STORAGE_BUFFER:
                header += self.backend.storage_buffer_declaration(emitted_binding, binding.dtype, binding.name)
                binding_type_list.append(binding.binding_type)
                binding_access.append((
                    self.binding_read_access[ii + 1],
                    self.binding_write_access[ii + 1]
                ))
            else:
                header += self.backend.sampler_declaration(emitted_binding, binding.dimension, binding.name)
                binding_type_list.append(binding.binding_type)
                binding_access.append((
                    self.binding_read_access[ii + 1],
                    self.binding_write_access[ii + 1]
                ))
        
        pc_elements = self.pc_struct.build()

        pc_decleration_contents = self.compose_struct_decleration(pc_elements)
        
        if len(pc_decleration_contents) > 0:
            assert self.backend.name != "cuda", "Push Constants are not supported for the CUDA backend"
            header += self.backend.push_constant_declaration(pc_decleration_contents)

        pre_header = self.backend.pre_header(
            enable_subgroup_ops=not (self.flags & ShaderFlags.NO_SUBGROUP_OPS),
            enable_printf=not (self.flags & ShaderFlags.NO_PRINTF)
        )

        return ShaderDescription(
            header=f"{pre_header}{header}",
            body=self.backend.entry_point(self.contents),
            name=name,
            pc_size=self.pc_struct.size, 
            pc_structure=pc_elements, 
            uniform_structure=uniform_elements, 
            binding_type_list=[binding.value for binding in binding_type_list],
            binding_access=binding_access,
            exec_count_name=self.exec_count.raw_name if self.exec_count is not None else None,
            resource_binding_base=binding_base,
            backend=self.backend
        )
