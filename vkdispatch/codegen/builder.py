import vkdispatch.base.dtype as dtypes

from .struct_builder import StructElement, StructBuilder

from .shader_writer import ShaderWriter
from .backends import CodeGenBackend
from .global_builder import get_codegen_backend

from enum import IntFlag, auto

from typing import Dict, List, Optional, Any, get_type_hints
import dataclasses

from ..base.dtype import is_dtype
from .arguments import Constant, Variable, Buffer
from .variables.variables import BaseVariable, ShaderVariable, ScaledAndOfftsetIntVariable
from .variables.bound_variables import BufferVariable, ImageVariable

from .shader_description import ShaderDescription, BindingType, ShaderArgumentInfo, ShaderArgumentType

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

def annotation_to_shader_arg_and_variable(builder: "ShaderBuilder", type_annotation: Any, name: str, default_value: Any):
    # Dataclass case
    if(dataclasses.is_dataclass(type_annotation)):
        creation_args: Dict[str, ShaderVariable] = {}
        value_name = {}

        for field_name, field_type in get_type_hints(type_annotation).items():
            assert is_dtype(field_type), f"Unsupported type '{field_type}' for field '{type_annotation}.{field_name}'"

            creation_args[field_name] = builder._declare_constant(field_type)
            value_name[field_name] = creation_args[field_name].raw_name

        return ShaderArgumentInfo(
            name,
            ShaderArgumentType.CONSTANT_DATACLASS,
            default_value,
            value_name
        ), type_annotation(**creation_args)
    
    # Buffer case
    if(issubclass(type_annotation.__origin__, Buffer)):
        shader_var = builder._declare_buffer(type_annotation.__args__[0])

        return ShaderArgumentInfo(
            name,
            ShaderArgumentType.BUFFER,
            default_value,
            shader_var.raw_name,
            shader_shape_name=shader_var.shape_name,
            binding=shader_var.binding
        ), shader_var

    # Image case
    if(issubclass(type_annotation.__origin__, ImageVariable)):
        shader_var = builder._declare_image(
            type_annotation.__origin__.dimensions
        )

        return ShaderArgumentInfo(
            name,
            ShaderArgumentType.IMAGE,
            default_value,
            shader_var.raw_name,
            binding=shader_var.binding
        ), shader_var

    if(issubclass(type_annotation.__origin__, Constant)):
        shader_var = builder._declare_constant(type_annotation.__args__[0])

        return ShaderArgumentInfo(
            name,
            ShaderArgumentType.CONSTANT,
            default_value,
            shader_var.raw_name
        ), shader_var

    if(issubclass(type_annotation.__origin__, Variable)):
        shader_var = builder._declare_variable(type_annotation.__args__[0])

        return ShaderArgumentInfo(
            name,
            ShaderArgumentType.VARIABLE,
            default_value,
            shader_var.raw_name
        ), shader_var

    raise ValueError(f"Unsupported type '{type_annotation.__args__[0]}'")

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
    shader_arg_infos: List[ShaderArgumentInfo]
    shader_args: List[ShaderVariable]
    has_ubo: bool

    def __init__(self, flags: ShaderFlags = ShaderFlags.NONE):
        super().__init__()

        self.flags = flags
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
        self.shader_arg_infos = []
        self.shader_args = []
        self.has_ubo = False
        
        self.exec_count = None

        if not (self.flags & ShaderFlags.NO_EXEC_BOUNDS):
            self.exec_count = self._declare_constant(dtypes.uvec4, var_name="exec_count")
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

    def declare_shader_arguments(self,
                                 type_annotations: List,
                                 names: Optional[List[str]] = None,
                                 defaults: Optional[List[Any]] = None):
        assert len(self.shader_args) == 0, "Shader arguments have already been declared for this builder instance"

        for i in range(len(type_annotations)):
            shader_arg_info, shader_var = annotation_to_shader_arg_and_variable(
                self,
                type_annotations[i],
                names[i] if names is not None else f"param{i}",
                defaults[i] if defaults is not None else None
            )

            self.shader_args.append(shader_var)
            self.shader_arg_infos.append(shader_arg_info)

    def get_shader_arguments(self) -> List[ShaderVariable]:
        return self.shader_args

    def _declare_constant(self, var_type: dtypes.dtype, count: int = 1, var_name: Optional[str] = None):
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

    def _declare_variable(self, var_type: dtypes.dtype, count: int = 1, var_name: Optional[str] = None):
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
    
    def _declare_buffer(self, var_type: dtypes.dtype, var_name: Optional[str] = None):
        buffer_name = f"buf{self.binding_count}" if var_name is None else var_name
        shape_name = f"{buffer_name}_shape"
        scalar_expr = None

        if self.backend.name == "opencl" and (dtypes.is_vector(var_type) or dtypes.is_complex(var_type)):
            scalar_expr = f"{buffer_name}_scalar"
        
        self.binding_list.append(ShaderBinding(var_type, buffer_name, 0, BindingType.STORAGE_BUFFER))
        self.binding_read_access[self.binding_count] = False
        self.binding_write_access[self.binding_count] = False

        current_binding_count = self.binding_count

        def read_lambda():
            self.binding_read_access[current_binding_count] = True

        def write_lambda():
            self.binding_write_access[current_binding_count] = True

        def shape_var_factory():
            return self._declare_constant(dtypes.ivec4, var_name=shape_name)
        
        self.binding_count += 1
        
        return BufferVariable(
            var_type,
            self.binding_count-1,
            f"{buffer_name}.data",
            shape_var_factory=shape_var_factory,
            shape_name=shape_name,
            scalar_expr=scalar_expr,
            codegen_backend=self.backend,
            read_lambda=read_lambda,
            write_lambda=write_lambda
        )
    
    def _declare_image(self, dimensions: int, var_name: Optional[str] = None):
        image_name = f"tex{self.binding_count}" if var_name is None else var_name
        self.binding_list.append(ShaderBinding(dtypes.vec4, image_name, dimensions, BindingType.SAMPLER))
        self.binding_read_access[self.binding_count] = False
        self.binding_write_access[self.binding_count] = False

        current_binding_count = self.binding_count

        def read_lambda():
            self.binding_read_access[current_binding_count] = True

        def write_lambda():
            self.binding_write_access[current_binding_count] = True

        self.binding_count += 1
        
        return ImageVariable(
            dtypes.vec4,
            self.binding_count-1,
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
            scalar_expr=None,
            codegen_backend=self.backend,
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

        uniform_elements = []
        binding_type_list: List[BindingType] = []
        binding_access = []
        binding_base = 0
        
        if not self.uniform_struct.empty():
            uniform_elements = self.uniform_struct.build()

            uniform_decleration_contents = self.compose_struct_decleration(uniform_elements)
            header += self.backend.uniform_block_declaration(uniform_decleration_contents)

            binding_type_list.append(BindingType.UNIFORM_BUFFER)
            binding_access.append((True, False))  # UBO is read-only
            binding_base = 1

            for shader_arg_info in self.shader_arg_infos:
                if (shader_arg_info.arg_type == ShaderArgumentType.BUFFER or
                    shader_arg_info.arg_type == ShaderArgumentType.IMAGE):
                    shader_arg_info.binding += 1 # Shift bindings by 1 to account for UBO at binding 0
        
        for ii, binding in enumerate(self.binding_list):
            if binding.binding_type == BindingType.STORAGE_BUFFER:
                header += self.backend.storage_buffer_declaration(
                    binding=ii + binding_base,
                    var_type=binding.dtype,
                    name=binding.name
                )
            else:
                header += self.backend.sampler_declaration(
                    binding=ii + binding_base,
                    dimensions=binding.dimension,
                    name=binding.name
                )
            
            binding_type_list.append(binding.binding_type)
            binding_access.append((
                self.binding_read_access[ii],
                self.binding_write_access[ii]
            ))
        
        pc_elements = self.pc_struct.build()

        pc_decleration_contents = self.compose_struct_decleration(pc_elements)
        
        if len(pc_decleration_contents) > 0:
            header += self.backend.push_constant_declaration(pc_decleration_contents)

        enable_subgroup_ops = (
            not (self.flags & ShaderFlags.NO_SUBGROUP_OPS)
            and self.backend.uses_feature("subgroup_ops")
        )
        enable_printf = (
            not (self.flags & ShaderFlags.NO_PRINTF)
            and self.backend.uses_feature("printf")
        )

        pre_header = self.backend.pre_header(
            enable_subgroup_ops=enable_subgroup_ops,
            enable_printf=enable_printf,
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
            backend=self.backend,
            shader_arg_infos=self.shader_arg_infos
        )
