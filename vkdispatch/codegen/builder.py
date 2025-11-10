import vkdispatch.base.dtype as dtypes
from vkdispatch.base.dtype import dtype

from .struct_builder import StructElement, StructBuilder

from .shader_writer import ShaderWriter

from enum import IntFlag, auto

from typing import Dict
from typing import List
from typing import Union
from typing import Optional

import dataclasses

from .variables.variables import BaseVariable, ShaderVariable, var_types_to_floating, SharedBuffer, BindingType, ShaderDescription, ScaledAndOfftsetIntVariable
from .variables.bound_variables import BufferVariable, ImageVariable


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
    dtype: dtype
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
    pre_header: str
    flags: ShaderFlags

    def __init__(self, flags: ShaderFlags = ShaderFlags.NONE, is_apple_device: bool = False) -> None:
        super().__init__()

        self.flags = flags
        self.is_apple_device = is_apple_device

        self.pre_header = "#version 450\n"
        self.pre_header += "#extension GL_ARB_separate_shader_objects : require\n"
        self.pre_header += "#extension GL_EXT_scalar_block_layout : require\n"

        if not (self.flags & ShaderFlags.NO_SUBGROUP_OPS):
            self.pre_header += "#extension GL_KHR_shader_subgroup_arithmetic : require\n"

        if not (self.flags & ShaderFlags.NO_PRINTF):
            self.pre_header += "#extension GL_EXT_debug_printf : require\n"
        
        self.reset()

    def reset(self) -> None:
        self.binding_count = 0
        self.pc_struct = StructBuilder()
        self.uniform_struct = StructBuilder()
        self.binding_list = []
        self.binding_read_access = {}
        self.binding_write_access = {}
        self.shared_buffers = []
        self.scope_num = 1
        # self.mapping_index: ShaderVariable = None
        # self.kernel_index: ShaderVariable = None
        # self.mapping_registers: List[ShaderVariable] = None
        
        self.exec_count = self.declare_constant(dtypes.uvec4, var_name="exec_count")
        
        if not (self.flags & ShaderFlags.NO_EXEC_BOUNDS):
            self.append_contents(
                f"if(any(lessThanEqual({self.exec_count.resolve()}.xyz, gl_GlobalInvocationID))) {{ return; }}"
            )

    def new_var(self,
                var_type: dtype,
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

    def declare_constant(self, var_type: dtype, count: int = 1, var_name: Optional[str] = None):
        if var_name is None:
            var_name = self.new_name()

        new_var = ShaderVariable(
            var_type=var_type,
            name=f"UBO.{var_name}",
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

    def declare_variable(self, var_type: dtype, count: int = 1, var_name: Optional[str] = None):
        if var_name is None:
            var_name = self.new_name()

        new_var = ShaderVariable(
            var_type=var_type,
            name=f"PC.{var_name}",
            raw_name=var_name,
            lexical_unit=True,
            settable=False,
            parents=[]
        )

        new_var._varying = True

        if count > 1:
            new_var.use_child_type = False
            new_var.can_index = True

        self.pc_struct.register_element(new_var.raw_name, var_type, count)
        return new_var
    
    def declare_buffer(self, var_type: dtype, var_name: Optional[str] = None):
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
        
        return BufferVariable(
            var_type,
            self.binding_count,
            f"{buffer_name}.data",
            self.declare_constant(dtypes.ivec4, var_name=shape_name),
            shape_name,
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
    
    def shared_buffer(self, var_type: dtype, size: int, var_name: Optional[str] = None):
        if var_name is None:
            var_name = self.new_name()
        
        shape_name = f"{var_name}_shape"

        new_var = BufferVariable(
            var_type,
            -1,
            var_name,
            self.declare_constant(dtypes.ivec4, var_name=shape_name),
            shape_name,
            read_lambda=lambda: None,
            write_lambda=lambda: None
        )

        self.shared_buffers.append(SharedBuffer(var_type, size, new_var.name))

        return new_var
    
    def compose_struct_decleration(self, elements: List[StructElement]) -> str:
        declerations = []

        for elem in elements:
            decleration_type = f"{elem.dtype.glsl_type}"

            decleration_suffix = ""
            if elem.count > 1:
                decleration_suffix = f"[{elem.count}]"

            declerations.append(f"\t{decleration_type} {elem.name}{decleration_suffix};")
        
        return "\n".join(declerations)

    def build(self, name: str) -> ShaderDescription:
        header = "" + self.pre_header

        for shared_buffer in self.shared_buffers:
            header += f"shared {shared_buffer.dtype.glsl_type} {shared_buffer.name}[{shared_buffer.size}];\n"

        uniform_elements = self.uniform_struct.build()
        
        uniform_decleration_contents = self.compose_struct_decleration(uniform_elements)
        if len(uniform_decleration_contents) > 0:
            header += f"\nlayout(set = 0, binding = 0) uniform UniformObjectBuffer {{\n { uniform_decleration_contents } \n}} UBO;\n"

        binding_type_list = [BindingType.UNIFORM_BUFFER]
        binding_access = [(True, False)]  # UBO is read-only
        
        for ii, binding in enumerate(self.binding_list):
            if binding.binding_type == BindingType.STORAGE_BUFFER:
                true_type = binding.dtype.glsl_type

                header += f"layout(set = 0, binding = {ii + 1}) buffer Buffer{ii + 1} {{ {true_type} data[]; }} {binding.name};\n"
                binding_type_list.append(binding.binding_type)
                binding_access.append((
                    self.binding_read_access[ii + 1],
                    self.binding_write_access[ii + 1]
                ))
            else:
                header += f"layout(set = 0, binding = {ii + 1}) uniform sampler{binding.dimension}D {binding.name};\n"
                binding_type_list.append(binding.binding_type)
                binding_access.append((
                    self.binding_read_access[ii + 1],
                    self.binding_write_access[ii + 1]
                ))
        
        pc_elements = self.pc_struct.build()

        pc_decleration_contents = self.compose_struct_decleration(pc_elements)
        
        if len(pc_decleration_contents) > 0:
            header += f"\nlayout(push_constant) uniform PushConstant {{\n { pc_decleration_contents } \n}} PC;\n"

        return ShaderDescription(
            header=header,
            body=f"void main() {{\n{self.contents}\n}}\n",
            name=name,
            pc_size=self.pc_struct.size, 
            pc_structure=pc_elements, 
            uniform_structure=uniform_elements, 
            binding_type_list=[binding.value for binding in binding_type_list],
            binding_access=binding_access,
            exec_count_name=self.exec_count.raw_name
        )