import vkdispatch.base.dtype as dtypes
from vkdispatch.base.dtype import dtype, is_vector, is_matrix, is_complex, to_vector

from .struct_builder import StructElement, StructBuilder

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import Callable
from typing import Any

import enum
import dataclasses

import numpy as np

ENABLE_SCALED_AND_OFFSET_INT = True

def do_scaled_int_check(other):
    return ENABLE_SCALED_AND_OFFSET_INT and (isinstance(other, int) or np.issubdtype(type(other), np.integer))

def is_int_power_of_2(n: int) -> bool:
    """Check if an integer is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0

def shader_var_name(index: "Union[Any, ShaderVariable]") -> str:
    if isinstance(index, ShaderVariable):
        result_str = str(index)

        if result_str[0] == "(" and result_str[-1] == ")":
            result_str = result_str[1:-1]
        
        return result_str
    
    return str(index)

def var_types_to_floating(var_type: dtype) -> dtype:
    if var_type == dtypes.int32 or var_type == dtypes.uint32:
        return dtypes.float32

    if var_type == dtypes.ivec2 or var_type == dtypes.uvec2:
        return dtypes.vec2

    if var_type == dtypes.ivec3 or var_type == dtypes.uvec3:
        return dtypes.vec3
    
    if var_type == dtypes.ivec4 or var_type == dtypes.uvec4:
        return dtypes.vec4
    
    return var_type

class BindingType(enum.Enum):
    """
    A dataclass that represents the type of a binding in a shader. Either a
    STORAGE_BUFFER, UNIFORM_BUFFER, or SAMPLER.
    """
    STORAGE_BUFFER = 1
    UNIFORM_BUFFER = 3
    SAMPLER = 5

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

@dataclasses.dataclass
class SharedBuffer:
    """
    A dataclass that represents a shared buffer in a shader.

    Attributes:
        dtype (vd.dtype): The dtype of the shared buffer.
        size (int): The size of the shared buffer.
        name (str): The name of the shared buffer within the shader code.
    """
    dtype: dtype
    size: int
    name: str

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
    exec_count_name: str

    def make_source(self, x: int, y: int, z: int) -> str:
        layout_str = f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;"
        return f"{self.header}\n{layout_str}\n{self.body}"

class ShaderVariable:
    append_func: Callable[[str], None]
    name_func: Callable[[str], str]
    var_type: dtype
    name: str
    raw_name: str
    can_index: bool = False
    use_child_type: bool = True
    index_suffix: str = ""
    _varying: bool = False
    lexical_unit: bool = False
    settable: bool = False

    def __init__(self, 
                 append_func: Callable[[str], None], 
                 name_func: Callable[[str], Tuple[str, str]], 
                 var_type: dtype, 
                 name: Optional[str] = None,
                 lexical_unit: bool = False,
                 settable: bool = False
        ) -> None:

        self.append_func = append_func
        self.name_func = name_func
        self.var_type = var_type
        self.lexical_unit = lexical_unit

        both_names = self.name_func(name)
        self.name = both_names[0]
        self.raw_name = both_names[1]
        self.settable = settable

        if is_complex(self.var_type):
            self.real = self.new(self.var_type.child_type, f"{self}.x", lexical_unit=True, settable=settable)
            self.imag = self.new(self.var_type.child_type, f"{self}.y", lexical_unit=True, settable=settable)
            self.x = self.real
            self.y = self.imag

            self._register_shape()
        
        if is_vector(self.var_type):
            self.x = self.new(self.var_type.child_type, f"{self}.x", lexical_unit=True, settable=settable)
            
            if self.var_type.child_count >= 2:
                self.y = self.new(self.var_type.child_type, f"{self}.y", lexical_unit=True, settable=settable)

            if self.var_type.child_count >= 3:
                self.z = self.new(self.var_type.child_type, f"{self}.z", lexical_unit=True, settable=settable)

            if self.var_type.child_count == 4:
                self.w = self.new(self.var_type.child_type, f"{self}.w", lexical_unit=True, settable=settable)
            
            self._register_shape()
        
        if is_matrix(self.var_type):
            self._register_shape()

        self._initilized = True

    def __repr__(self) -> str:
        if self.lexical_unit:
            return self.name

        return f"({self.name})"

    def new(self, var_type: dtype, name: str = None, lexical_unit: bool = False, settable: bool = False):
        return ShaderVariable(self.append_func, self.name_func, var_type, name, lexical_unit=lexical_unit, settable=settable)
       
    def __getitem__(self, index) -> "ShaderVariable":
        if not self.can_index:
            raise ValueError("Unsupported indexing!")
        
        return_type = self.var_type.child_type if self.use_child_type else self.var_type

        if isinstance(index, ShaderVariable) or isinstance(index, (int, np.integer)):
            return self.new(return_type, f"{self.name}[{shader_var_name(index)}]", settable=self.settable)
        
        if isinstance(index, tuple):
            index_strs = tuple(shader_var_name(i) for i in index)

            if len(index_strs) == 1:
                return self.new(return_type, f"{self.name}[{index_strs[0]}]{self.index_suffix}", settable=self.settable)
            elif self.shape is None:
                raise ValueError("Cannot do multidimentional index into object with no shape!")
            
            if len(index_strs) == 2:
                true_index = f"{index_strs[0]} * {self.shape.y} + {index_strs[1]}"
                return self.new(return_type, f"{self.name}[{true_index}]{self.index_suffix}", settable=self.settable)
            elif len(index_strs) == 3:
                true_index = f"{index_strs[0]} * {self.shape.y} + {index_strs[1]}"
                true_index = f"({true_index}) * {self.shape.z} + {index_strs[2]}"
                return self.new(return_type, f"{self.name}[{true_index}]{self.index_suffix}", settable=self.settable)
            else:
                raise ValueError(f"Unsupported number of indicies {len(index)}!")

        else:
            raise ValueError(f"Unsupported index type {index} of type {type(index)}!")

    def __setitem__(self, index, value: "ShaderVariable") -> None:
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        if isinstance(index, slice):
            if index.start is None and index.stop is None and index.step is None:
                self.append_func(f"{self.name} = {shader_var_name(value)};\n")
                return
            else:
                raise ValueError("Unsupported slice!")

        if not self.can_index:
            raise ValueError(f"Unsupported indexing {index}!")
        
        if f"{self.name}[{index}]{self.index_suffix}" == str(value):
            return

        self.append_func(f"{self.name}[{shader_var_name(index)}]{self.index_suffix} = {shader_var_name(value)};\n")

    def _register_shape(self, shape_var: "ShaderVariable" = None, shape_name: str = None, use_child_type: bool = True):
        self.shape = shape_var
        self.shape_name = shape_name
        self.can_index = True
        self.use_child_type = use_child_type

    def __bool__(self) -> bool:
        raise ValueError(f"Vkdispatch variables cannot be cast to a python boolean")
 
    def new_scaled_and_offset_int(self, var_type: dtype, name: str = None):
        return ScaledAndOfftsetIntVariable(self.append_func, self.name_func, var_type, name)

    def copy(self, var_name: str = None):
        """Create a new variable with the same value as the current variable."""
        new_var = self.new(self.var_type, var_name, lexical_unit=True, settable=True)
        self.append_func(f"{self.var_type.glsl_type} {new_var.name} = {self};\n")
        return new_var

    def cast_to(self, var_type: dtype):
        return self.new(var_type, f"{var_type.glsl_type}({self.name})", lexical_unit=True)

    def printf_args(self) -> str:
        total_count = np.prod(self.var_type.shape)

        if total_count == 1:
            return self.name

        args_list = []

        for i in range(0, total_count):
            args_list.append(f"{self.name}[{i}]")

        return ",".join(args_list)

    def __setattr__(self, name: str, value: "ShaderVariable") -> "ShaderVariable":
        try:
            if self._initilized:
                if is_complex(self.var_type): #.structure == vd.dtype_structure.DATA_STRUCTURE_SCALAR and self.var_type.is_complex:
                    if name == "real":
                        self.append_func(f"{self}.x = {shader_var_name(value)};\n")
                        return
                    
                    if name == "imag":
                        self.append_func(f"{self}.y = {shader_var_name(value)};\n")
                        return
                
                    if name == "x" or name == "y":
                        self.append_func(f"{self}.{name} = {shader_var_name(value)};\n")
                        return
                
                if is_vector(self.var_type):#self.var_type.structure == vd.dtype_structure.DATA_STRUCTURE_VECTOR:
                    if name == "x" or name == "y" or name == "z" or name == "w":
                        self.append_func(f"{self}.{name} = {shader_var_name(value)};\n")
                        return
        except:
            super().__setattr__(name, value)
            return
        

        super().__setattr__(name, value)
        
        #if hasattr(self, name):
        #    super().__setattr__(name, value)
        #    return

        #raise AttributeError(f"Cannot set attribute '{name}'")

    def __getattr__(self, name: str) -> "ShaderVariable":
        if not set(name).issubset(set("xyzw")):
            raise AttributeError(f"Cannot get attribute '{name}'")

        if len(name) > 4:
            raise AttributeError(f"Cannot get attribute '{name}'")
        
        if len(name) == 1:
            if len(self.var_type.shape) == 2:
                raise AttributeError(f"Cannot get attribute '{name}' from a matrix of shape {self.var_type.shape}!")
            
            if name == "x" and self.var_type.shape[0] == 1:
                return self.new(self.var_type, f"{self}.{name}", lexical_unit=True)
            
            if name == "y" and self.var_type.shape[0] < 2:
                raise AttributeError(f"Cannot get attribute '{name}' from a {self.var_type.name}!")
            
            if name == "z" and self.var_type.shape[0] < 3:
                raise AttributeError(f"Cannot get attribute '{name}' from a {self.var_type.name}!")

            if name == "w" and self.var_type.shape[0] < 4:
                raise AttributeError(f"Cannot get attribute '{name}' from a {self.var_type.name}!")

            return self.new(self.var_type.child_type, f"{self}.{name}", lexical_unit=True)
        
        new_type = to_vector(self.var_type.child_type, len(name))
        return self.new(new_type, f"{self}.{name}", lexical_unit=True)

    def __lt__(self, other):
        return self.new(dtypes.int32, f"{self} < {other}")

    def __le__(self, other):
        return self.new(dtypes.int32, f"{self} <= {other}")

    def __eq__(self, other):
        return self.new(dtypes.int32, f"{self} == {other}")

    def __ne__(self, other):
        return self.new(dtypes.int32, f"{self} != {other}")

    def __gt__(self, other):
        return self.new(dtypes.int32, f"{self} > {other}")

    def __ge__(self, other):
        return self.new(dtypes.int32, f"{self} >= {other}")

    def __add__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}")
            return result.__add__(other)

        return self.new(self.var_type, f"{self} + {other}")

    def __sub__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}")
            return result.__sub__(other)
        
        return self.new(self.var_type, f"{self} - {other}")

    def __mul__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}")
            return result.__mul__(other)

        return_var_type = self.var_type

        if (self.var_type.dimentions == 2
            and other.var_type.dimentions == 1):
            return_var_type = other.var_type

        return self.new(return_var_type, f"{self} * {other}")

    def __truediv__(self, other):
        # if do_scaled_int_check(other) and is_int_power_of_2(other):
        #     if other == 1:
        #         return self
            
        #     if self.var_type != vd.int32 or self.var_type != vd.uint32:
        #         return self.new(self.var_type, f"{self} / {other}")

        #     power = int(np.round(np.log2(other)))

        #     return self.new(self.var_type, f"({self} >> {power})")

        return self.new(self.var_type, f"{self} / {other}")

    # def __floordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    return self.builder.make_var(f"{self} / {other}")

    def __mod__(self, other):
        return self.new(self.var_type, f"{self} % {other}")

    def __pow__(self, other):
        other_str = str(other)

        if isinstance(other, ShaderVariable):
            other_str = other.name

        return self.new(self.var_type, f"pow({self.name}, {other_str})")

    def __neg__(self):
        return self.new(self.var_type, f"-{self}")

    def __abs__(self):
        return self.new(self.var_type, f"abs({self.name})")

    def __invert__(self):
        return self.new(self.var_type, f"~{self}")

    def __lshift__(self, other):
        return self.new(self.var_type, f"{self} << {other}")

    def __rshift__(self, other):
        return self.new(self.var_type, f"{self} >> {other}")

    def __and__(self, other):
        return self.new(self.var_type, f"{self} & {other}")

    def __xor__(self, other):
        return self.new(self.var_type, f"{self} ^ {other}")

    def __or__(self, other):
        return self.new(self.var_type, f"({self} | {other}")

    def __radd__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}")
            return result.__radd__(other)

        return self.new(self.var_type, f"{other} + {self}")

    def __rsub__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}")
            return result.__rsub__(other)

        return self.new(self.var_type, f"{other} - {self}")

    def __rmul__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}")
            return result.__rmul__(other)

        return self.new(self.var_type, f"{other} * {self}")

    def __rtruediv__(self, other):
        return self.new(self.var_type, f"{other} / {self}")

    # def __rfloordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    return self.builder.make_var(f"{other} / {self}")

    def __rmod__(self, other):
        return self.new(self.var_type, f"{other} % {self}")

    def __rpow__(self, other):
        other_str = str(other)

        if isinstance(other, ShaderVariable):
            other_str = other.name

        return self.new(self.var_type, f"pow({other_str}, {self.name})")

    def __rand__(self, other):
        return self.new(self.var_type, f"{other} & {self}")

    def __rxor__(self, other):
        return self.new(self.var_type, f"{other} ^ {self}")

    def __ror__(self, other):
        return self.new(self.var_type, f"{other} | {self}")

    def __iadd__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.append_func(f"{self} += {other};\n")
        return self

    def __isub__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.append_func(f"{self} -= {other};\n")
        return self

    def __imul__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.append_func(f"{self} *= {other};\n")
        return self

    def __itruediv__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.append_func(f"{self} /= {other};\n")
        return self

    # def __ifloordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    self.append_func(f"{self} /= {other};\n")
    #    return self

    def __imod__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.append_func(f"{self} %= {other};\n")
        return self

    def __ipow__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        other_str = str(other)

        if isinstance(other, ShaderVariable):
            other_str = other.name

        self.append_func(f"{self} = pow({self.name}, {other_str});\n")
        return self

    def __ilshift__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.append_func(f"{self} <<= {other};\n")
        return self

    def __irshift__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.append_func(f"{self} >>= {other};\n")
        return self

    def __iand__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.append_func(f"{self} &= {other};\n")
        return self

    def __ixor__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.append_func(f"{self} ^= {other};\n")
        return self

    def __ior__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.append_func(f"{self} |= {other};\n")
        return self

class ScaledAndOfftsetIntVariable(ShaderVariable):
    def __init__(self, 
                 append_func: Callable[[str], None], 
                 name_func: Callable[[str], Tuple[str, str]], 
                 var_type: dtype, 
                 name: Optional[str] = None,
                 scale: int = 1,
                 offset: int = 0
        ) -> None:
        self.base_name = str(name)
        self.scale = scale
        self.offset = offset
        
        super().__init__(append_func, name_func, var_type, name)
    
    def new_from_self(self, scale: int = 1, offset: int = 0):
        return ScaledAndOfftsetIntVariable(
            self.append_func,
            self.name_func,
            self.var_type,
            f"{self.name}",
            scale=self.scale * scale,
            offset=offset + self.offset * scale
        )

    def __repr__(self) -> str:
        scale_str = f" * {self.scale}" if self.scale != 1 else ""
        offset_str = f" + {self.offset}" if self.offset != 0 else ""

        if scale_str == "" and offset_str == "":
            return self.base_name

        return f"({self.base_name}{scale_str}{offset_str})"

    def __add__(self, other):
        if isinstance(other, ShaderVariable):
            return super().__add__(other)

        return self.new_from_self(offset=other)

    def __sub__(self, other):
        if isinstance(other, ShaderVariable):
            return super().__sub__(other)

        return self.new_from_self(offset=-other)

    def __mul__(self, other):
        if isinstance(other, ShaderVariable):
            return super().__mul__(other)

        return self.new_from_self(scale=other)
    
    def __radd__(self, other):
        if isinstance(other, ShaderVariable):
            return super().__radd__(other)

        return self.new_from_self(offset=other)

    def __rsub__(self, other):
        if isinstance(other, ShaderVariable):
            return super().__rsub__(other)

        return self.new_from_self(offset=other, scale=-1)

    def __rmul__(self, other):
        if isinstance(other, ShaderVariable):
            return super().__rmul__(other)

        return self.new_from_self(scale=other)

class BoundVariable(ShaderVariable):
    binding: int = -1

    def __init__(self,
                 append_func: Callable[[str], None],
                 name_func: Callable[[str], str],
                 var_type: dtype,
                 binding: int,
                 name: Optional[str] = None,
            ) -> None:
            super().__init__(append_func, name_func, var_type, name)

            self.binding = binding
    
    #def __int__(self):
    #    return int(self.binding)

class BufferVariable(BoundVariable):
    def __init__(self,
                 append_func: Callable[[str], None],
                 name_func: Callable[[str], Tuple[str, str]], 
                 var_type: dtype,
                 binding: int,
                 name: Optional[str] = None,
                 shape_var: "ShaderVariable" = None,
                 shape_name: Optional[str] = None,
                 raw_name: Optional[str] = None
            ) -> None:
            super().__init__(append_func, name_func, var_type, binding, name)

            self.name = name if name is not None else self.name
            self.raw_name = raw_name if raw_name is not None else self.raw_name
            self.settable = True

            self._register_shape(shape_var=shape_var, shape_name=shape_name, use_child_type=False)

class ImageVariable(BoundVariable):
    dimensions: int = 0

    def __init__(self,
                 append_func: Callable[[str], None],
                 name_func: Callable[[str], Tuple[str, str]], 
                 var_type: dtype,
                 binding: int,
                 dimensions: int,
                 name: Optional[str] = None,
            ) -> None:
            super().__init__(append_func, name_func, var_type, binding, name)

            self.dimensions = dimensions
    
    def sample(self, coord: "ShaderVariable", lod: "ShaderVariable" = None) -> "ShaderVariable":
        if self.dimensions == 0:
            raise ValueError("Cannot sample a texture with dimension 0!")
        
        sample_coord_string = ""

        if self.dimensions == 1:
            sample_coord_string = f"((({coord}) + 0.5) / textureSize({self}, 0))"        
        elif self.dimensions == 2:
            sample_coord_string = f"((vec2({coord}.xy) + 0.5) / vec2(textureSize({self}, 0)))"
        elif self.dimensions == 3:
            sample_coord_string = f"((vec3({coord}.xyz) + 0.5) / vec3(textureSize({self}, 0)))"
        else:
            raise ValueError("Unsupported number of dimensions!")

        if lod is None:
            return self.new(dtypes.v4, f"texture({self}, {sample_coord_string})")
        
        return self.new(dtypes.v4, f"textureLod({self}, {sample_coord_string}, {lod})")

class ShaderBuilder:
    var_count: int
    binding_count: int
    binding_read_access: Dict[int, bool]
    binding_write_access: Dict[int, bool]
    binding_list: List[ShaderBinding]
    shared_buffers: List[SharedBuffer]
    scope_num: int
    pc_struct: StructBuilder
    uniform_struct: StructBuilder
    exec_count: Optional[ShaderVariable]
    contents: str
    pre_header: str

    def __init__(self, enable_subgroup_ops: bool = True, enable_atomic_float_ops: bool = True, enable_printf: bool = True, enable_exec_bounds: bool = True) -> None:
        self.enable_subgroup_ops = enable_subgroup_ops
        self.enable_atomic_float_ops = enable_atomic_float_ops
        self.enable_printf = enable_printf
        self.enable_exec_bounds = enable_exec_bounds

        self.pre_header = "#version 450\n"
        self.pre_header += "#extension GL_ARB_separate_shader_objects : enable\n"

        if self.enable_subgroup_ops:
            self.pre_header += "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"
        
        #if self.enable_atomic_float_ops:
        #    self.pre_header += "#extension GL_EXT_shader_atomic_float : enable\n"

        if self.enable_printf:
            self.pre_header += "#extension GL_EXT_debug_printf : enable\n"
        
        self.global_invocation = self.make_var(dtypes.uvec3, "gl_GlobalInvocationID", lexical_unit=True)
        self.local_invocation = self.make_var(dtypes.uvec3, "gl_LocalInvocationID", lexical_unit=True)
        self.workgroup = self.make_var(dtypes.uvec3, "gl_WorkGroupID", lexical_unit=True)
        self.workgroup_size = self.make_var(dtypes.uvec3, "gl_WorkGroupSize", lexical_unit=True)
        self.num_workgroups = self.make_var(dtypes.uvec3, "gl_NumWorkGroups", lexical_unit=True)

        self.num_subgroups = self.make_var(dtypes.uint32, "gl_NumSubgroups", lexical_unit=True)
        self.subgroup_id = self.make_var(dtypes.uint32, "gl_SubgroupID", lexical_unit=True)

        self.subgroup_size = self.make_var(dtypes.uint32, "gl_SubgroupSize", lexical_unit=True)
        self.subgroup_invocation = self.make_var(dtypes.uint32, "gl_SubgroupInvocationID", lexical_unit=True)
        
        self.reset()

    def reset(self) -> None:
        self.var_count = 0
        self.binding_count = 0
        self.pc_struct = StructBuilder()
        self.uniform_struct = StructBuilder()
        self.binding_list = []
        self.binding_read_access = {}
        self.binding_write_access = {}
        self.shared_buffers = []
        self.scope_num = 1
        self.contents = ""
        self.mapping_index: ShaderVariable = None
        self.kernel_index: ShaderVariable = None
        self.mapping_registers: List[ShaderVariable] = None
        
        self.exec_count = self.declare_constant(dtypes.uvec4, var_name="exec_count")
        
        if self.enable_exec_bounds:
            self.if_statement(self.exec_count.x <= self.global_invocation.x)
            self.return_statement()
            self.end()

            self.if_statement(self.exec_count.y <= self.global_invocation.y)
            self.return_statement()
            self.end()

            self.if_statement(self.exec_count.z <= self.global_invocation.z)
            self.return_statement()
            self.end()

    def set_mapping_index(self, index: ShaderVariable):
        self.mapping_index = index

    def set_kernel_index(self, index: ShaderVariable):
        self.kernel_index = index

    def set_mapping_registers(self, registers: ShaderVariable):
        self.mapping_registers = list(registers)

    def append_contents(self, contents: str) -> None:
        self.contents += ("    " * self.scope_num) + contents

    def comment(self, comment: str) -> None:
        self.append_contents("\n")
        self.append_contents(f"/* {comment} */\n")
    
    def get_name_func(self, prefix: Optional[str] = None, suffix: Optional[str] = None):
        my_prefix = [prefix]
        my_suffix = [suffix]
        def get_name_val(var_name: Union[str, None] = None):
            new_var = f"var{self.var_count}" if var_name is None else var_name
            raw_name = new_var
            
            if var_name is None:
                self.var_count += 1

            if my_prefix[0] is not None:
                new_var = f"{my_prefix[0]}{new_var}"
                my_prefix[0] = None
            
            if my_suffix[0] is not None:
                new_var = f"{new_var}{my_suffix[0]}"
                my_suffix[0] = None

            return new_var, raw_name
        return get_name_val

    def make_var(self,
                 var_type: dtype,
                 var_name: Optional[str] = None,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 lexical_unit: bool = False,
                 settable: bool = False) -> ShaderVariable:
        return ShaderVariable(
            self.append_contents,
            self.get_name_func(prefix, suffix),
            var_type,
            var_name,
            lexical_unit=lexical_unit,
            settable=settable
        )
    
    def declare_constant(self, var_type: dtype, count: int = 1, var_name: Optional[str] = None):
        suffix = None
        if var_type.glsl_type_extern is not None:
            suffix = ".xyz"

        new_var = self.make_var(var_type, var_name, "UBO.", suffix)

        if count > 1:
            new_var.use_child_type = False
            new_var.can_index = True

        self.uniform_struct.register_element(new_var.raw_name, var_type, count)
        return new_var

    def declare_variable(self, var_type: dtype, count: int = 1, var_name: Optional[str] = None):
        suffix = None
        if var_type.glsl_type_extern is not None:
            suffix = ".xyz"
        
        new_var = self.make_var(var_type, var_name, "PC.", suffix)
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
        self.binding_read_access[self.binding_count] = True
        self.binding_write_access[self.binding_count] = True
        
        return BufferVariable(
            self.append_contents, 
            self.get_name_func(), 
            var_type,
            self.binding_count,
            f"{buffer_name}.data",
            self.declare_constant(dtypes.ivec4, var_name=shape_name),
            shape_name
        )
    
    def declare_image(self, dimensions: int, var_name: Optional[str] = None):
        self.binding_count += 1

        image_name = f"tex{self.binding_count}" if var_name is None else var_name
        self.binding_list.append(ShaderBinding(dtypes.vec4, image_name, dimensions, BindingType.SAMPLER))
        self.binding_read_access[self.binding_count] = False
        self.binding_write_access[self.binding_count] = False
        
        return ImageVariable(
            self.append_contents, 
            self.get_name_func(), 
            dtypes.vec4,
            self.binding_count,
            dimensions,
            f"{image_name}"
        )
    
    def shared_buffer(self, var_type: dtype, size: int, var_name: Optional[str] = None):
        buffer_name = self.get_name_func()(var_name)[0]
        shape_name = f"{buffer_name}_shape"

        new_var = BufferVariable(
            self.append_contents, 
            self.get_name_func(), 
            var_type,
            -1,
            buffer_name,
            self.declare_constant(dtypes.ivec4, var_name=shape_name),
            shape_name
        )

        self.shared_buffers.append(SharedBuffer(var_type, size, new_var.name))

        return new_var
    
    def abs(self, arg: ShaderVariable):
        return self.make_var(arg.var_type, f"abs({arg})", lexical_unit=True)
    
    def acos(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"acos({arg})", lexical_unit=True)

    def acosh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"acosh({arg})", lexical_unit=True)

    def asin(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"asin({arg})", lexical_unit=True)

    def asinh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"asinh({arg})", lexical_unit=True)

    def atan(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"atan({arg})", lexical_unit=True)
    
    def atan2(self, arg1: ShaderVariable, arg2: ShaderVariable):
        # TODO: correctly handle pure float inputs

        floating_arg1 = var_types_to_floating(arg1.var_type)
        floating_arg2 = var_types_to_floating(arg2.var_type)

        assert floating_arg1 == floating_arg2, f"Both arguments to atan2 ({arg1.var_type} and {arg2.var_type}) must be of the same dimentionality"

        return self.make_var(floating_arg1, f"atan({arg1}, {arg2})", lexical_unit=True)

    def atanh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"atanh({arg})", lexical_unit=True)
    
    def atomic_add(self, arg1: ShaderVariable, arg2: ShaderVariable):
        new_var = self.make_var(arg1.var_type)
        self.append_contents(f"{new_var.var_type.glsl_type} {new_var.name} = atomicAdd({arg1}, {arg2});\n")
        return new_var
    
    def barrier(self):
        self.append_contents("barrier();\n")
    
    def ceil(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"ceil({arg})", lexical_unit=True)
    
    def clamp(self, arg: ShaderVariable, min_val: ShaderVariable, max_val: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"clamp({arg}, {min_val}, {max_val})", lexical_unit=True)

    def cos(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"cos({arg})", lexical_unit=True)
    
    def cosh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"cosh({arg})", lexical_unit=True)
    
    def cross(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(dtypes.v3, f"cross({arg1}, {arg2})", lexical_unit=True)
    
    def degrees(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"degrees({arg})", lexical_unit=True)
    
    def determinant(self, arg: ShaderVariable):
        return self.make_var(dtypes.float32, f"determinant({arg})", lexical_unit=True)
    
    def distance(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(dtypes.float32, f"distance({arg1}, {arg2})", lexical_unit=True)
    
    def dot(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(dtypes.float32, f"dot({arg1}, {arg2})", lexical_unit=True)
    
    def exp(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"exp({arg})", lexical_unit=True)
    
    def exp2(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"exp2({arg})", lexical_unit=True)

    def float_bits_to_int(self, arg: ShaderVariable):
        return self.make_var(dtypes.int32, f"floatBitsToInt({arg})", lexical_unit=True)
    
    def float_bits_to_uint(self, arg: ShaderVariable):
        return self.make_var(dtypes.uint32, f"floatBitsToUint({arg})", lexical_unit=True)

    def floor(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"floor({arg})", lexical_unit=True)
    
    def fma(self, arg1: ShaderVariable, arg2: ShaderVariable, arg3: ShaderVariable):
        # TODO: properly handle type conversion and float inputs

        return self.make_var(arg1.var_type, f"fma({arg1}, {arg2}, {arg3})", lexical_unit=True)
    
    def int_bits_to_float(self, arg: ShaderVariable):
        return self.make_var(dtypes.float32, f"intBitsToFloat({arg})", lexical_unit=True)

    def inverse(self, arg: ShaderVariable):
        assert arg.var_type.dimentions == 2, f"Cannot apply inverse to non-matrix type {arg.var_type}"

        return self.make_var(arg.var_type, f"inverse({arg})", lexical_unit=True)
    
    def inverse_sqrt(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"inversesqrt({arg})", lexical_unit=True)
    
    def isinf(self, arg: ShaderVariable):
        return self.make_var(dtypes.int32, f"any(isinf({arg}))", lexical_unit=True)
    
    def isnan(self, arg: ShaderVariable):
        return self.make_var(dtypes.int32, f"any(isnan({arg}))", lexical_unit=True)

    def length(self, arg: ShaderVariable):
        return self.make_var(dtypes.float32, f"length({arg})", lexical_unit=True)

    def log(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"log({arg})", lexical_unit=True)

    def log2(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"log2({arg})", lexical_unit=True)

    def max(self, arg1: ShaderVariable, arg2: ShaderVariable):
        # TODO: properly handle type conversion and float inputs

        return self.make_var(arg1.var_type, f"max({arg1}, {arg2})", lexical_unit=True)

    def memory_barrier(self):
        self.append_contents("memoryBarrier();\n")

    def memory_barrier_shared(self):
        self.append_contents("memoryBarrierShared();\n")

    def min(self, arg1: ShaderVariable, arg2: ShaderVariable):
        # TODO: properly handle type conversion and float inputs

        return self.make_var(arg1.var_type, f"min({arg1}, {arg2})", lexical_unit=True)
    
    def mix(self, arg1: ShaderVariable, arg2: ShaderVariable, arg3: ShaderVariable):
        # TODO: properly handle type conversion and float inputs

        return self.make_var(arg1.var_type, f"mix({arg1}, {arg2}, {arg3})", lexical_unit=True)

    def mod(self, arg1: ShaderVariable, arg2: ShaderVariable):
        # TODO: properly handle type conversion and float inputs

        return self.make_var(arg1.var_type, f"mod({arg1}, {arg2})", lexical_unit=True)
    
    def normalize(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"normalize({arg})", lexical_unit=True)
    
    def pow(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(arg1.var_type, f"pow({arg1}, {arg2})", lexical_unit=True)
    
    def radians(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"radians({arg})", lexical_unit=True)
    
    def round(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"round({arg})", lexical_unit=True)
    
    def round_even(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"roundEven({arg})", lexical_unit=True)

    def sign(self, arg: ShaderVariable):
        return self.make_var(arg.var_type, f"sign({arg})", lexical_unit=True)

    def sin(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"sin({arg})", lexical_unit=True)
    
    def sinh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"sinh({arg})", lexical_unit=True)
    
    def smoothstep(self, arg1: ShaderVariable, arg2: ShaderVariable, arg3: ShaderVariable):
        return self.make_var(arg1.var_type, f"smoothstep({arg1}, {arg2}, {arg3})", lexical_unit=True)

    def sqrt(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"sqrt({arg})", lexical_unit=True)
    
    def step(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(arg1.var_type, f"step({arg1}, {arg2})", lexical_unit=True)

    def tan(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"tan({arg})", lexical_unit=True)
    
    def tanh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"tanh({arg})", lexical_unit=True)
    
    def transpose(self, arg: ShaderVariable):
        return self.make_var(arg.var_type, f"transpose({arg})", lexical_unit=True)
    
    def trunc(self, arg: ShaderVariable):
        return self.make_var(arg.var_type, f"trunc({arg})", lexical_unit=True)

    def uint_bits_to_float(self, arg: ShaderVariable):
        return self.make_var(dtypes.float32, f"uintBitsToFloat({arg})", lexical_unit=True)
    
    def mult_c64(self, arg1: ShaderVariable, arg2: ShaderVariable):
        new_var = self.make_var(arg1.var_type, f"vec2({arg1}.x * {arg2}.x - {arg1}.y * {arg2}.y, {arg1}.x * {arg2}.y + {arg1}.y * {arg2}.x)", lexical_unit=True)
        return new_var
    
    def mult_c64_by_const(self, arg1: ShaderVariable, number: complex):
        new_var = self.make_var(arg1.var_type, f"vec2({arg1}.x * {number.real} - {arg1}.y * {number.imag}, {arg1}.x * {number.imag} + {arg1}.y * {number.real})", lexical_unit=True)
        return new_var
    
    def mult_conj_c64(self, arg1: ShaderVariable, arg2: ShaderVariable):
        new_var = self.make_var(arg1.var_type, f"vec2({arg1}.x * {arg2}.x + {arg1}.y * {arg2}.y, {arg1}.y * {arg2}.x - {arg1}.x * {arg2}.y)", lexical_unit=True);
        return new_var

    def if_statement(self, arg: ShaderVariable, command: Optional[str] = None):
        if command is None:
            self.append_contents(f"if({arg}) {'{'}\n")
            self.scope_num += 1
            return
        
        self.append_contents(f"if({arg})\n")
        self.scope_num += 1
        self.append_contents(f"{command}\n")
        self.scope_num -= 1

    def if_any(self, *args: List[ShaderVariable]):
        self.append_contents(f"if({' || '.join([str(elem) for elem in args])}) {'{'}\n")
        self.scope_num += 1

    def if_all(self, *args: List[ShaderVariable]):
        self.append_contents(f"if({' && '.join([str(elem) for elem in args])}) {'{'}\n")
        self.scope_num += 1

    def else_statement(self):
        self.scope_num -= 1
        self.append_contents("} else {\n")
        self.scope_num += 1

    def else_if_statement(self, arg: ShaderVariable):
        self.scope_num -= 1
        self.append_contents(f"}} else if({arg}) {'{'}\n")
        self.scope_num += 1

    def else_if_any(self, *args: List[ShaderVariable]):
        self.scope_num -= 1
        self.append_contents(f"}} else if({' || '.join([str(elem) for elem in args])}) {'{'}\n")
        self.scope_num += 1
    
    def else_if_all(self, *args: List[ShaderVariable]):
        self.scope_num -= 1
        self.append_contents(f"}} else if({' && '.join([str(elem) for elem in args])}) {'{'}\n")
        self.scope_num += 1

    def return_statement(self, arg=None):
        arg = arg if arg is not None else ""
        self.append_contents(f"return {arg};\n")

    def while_statement(self, arg: ShaderVariable):
        self.append_contents(f"while({arg}) {'{'}\n")
        self.scope_num += 1

    def end(self):
        self.scope_num -= 1
        self.append_contents("}\n")

    def logical_and(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(dtypes.int32, f"({arg1} && {arg2})")

    def logical_or(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(dtypes.int32, f"({arg1} || {arg2})")

    def subgroup_add(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupAdd({arg1})", lexical_unit=True)

    def subgroup_mul(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMul({arg1})", lexical_unit=True)

    def subgroup_min(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMin({arg1})", lexical_unit=True)

    def subgroup_max(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMax({arg1})", lexical_unit=True)

    def subgroup_and(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupAnd({arg1})", lexical_unit=True)

    def subgroup_or(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupOr({arg1})", lexical_unit=True)

    def subgroup_xor(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupXor({arg1})", lexical_unit=True)

    def subgroup_elect(self):
        return self.make_var(dtypes.int32, f"subgroupElect()", lexical_unit=True)

    def subgroup_barrier(self):
        self.append_contents("subgroupBarrier();\n")

    def new(self, var_type: dtype, *args, var_name: Optional[str] = None):
        new_var = self.make_var(var_type, var_name=var_name, lexical_unit=True, settable=True)

        decleration_suffix = ""
        if len(args) > 0:
            decleration_suffix = f" = {var_type.glsl_type}({', '.join([str(elem) for elem in args])})"

        self.append_contents(f"{new_var.var_type.glsl_type} {new_var.name}{decleration_suffix};\n")

        return new_var

    def new_float(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.float32, *args, var_name=var_name)

    def new_int(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.int32, *args, var_name=var_name)

    def new_uint(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.uint32, *args, var_name=var_name)

    def new_vec2(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.vec2, *args, var_name=var_name)

    def new_vec3(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.vec3, *args, var_name=var_name)

    def new_vec4(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.vec4, *args, var_name=var_name)

    def new_uvec2(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.uvec2, *args, var_name=var_name)

    def new_uvec3(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.uvec3, *args, var_name=var_name)

    def new_uvec4(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.uvec4, *args, var_name=var_name)

    def new_ivec2(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.ivec2, *args, var_name=var_name)

    def new_ivec3(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.ivec3, *args, var_name=var_name)

    def new_ivec4(self, *args, var_name: Optional[str] = None):
        return self.new(dtypes.ivec4, *args, var_name=var_name)

    def printf(self, format: str, *args: Union[ShaderVariable, str], seperator=" "):
        args_string = ""

        for arg in args:
            args_string += f", {arg}"

        self.append_contents(f'debugPrintfEXT("{format}" {args_string});\n')

    def print_vars(self, *args: Union[ShaderVariable, str], seperator=" "):
        args_list = []

        fmts = []

        for arg in args:
            if isinstance(arg, ShaderVariable):
                args_list.append(arg.printf_args())
                fmts.append(arg.var_type.format_str)
            else:
                fmts.append(str(arg))

        fmt = seperator.join(fmts)
        
        args_argument = ""

        if len(args_list) > 0:
            args_argument = f", {','.join(args_list)}"

        self.append_contents(f'debugPrintfEXT("{fmt}"{args_argument});\n')

    def unravel_index(self, index: ShaderVariable, shape: ShaderVariable):
        new_var = self.new_uvec3()

        new_var.x = index % shape.x
        new_var.y = (index / shape.x) % shape.y
        new_var.z = index / (shape.x * shape.y)

        return new_var
    
    def complex_from_euler_angle(self, angle: ShaderVariable):
        return self.make_var(dtypes.vec2, f"vec2({self.cos(angle)}, {self.sin(angle)})")

    def compose_struct_decleration(self, elements: List[StructElement]) -> str:
        declerations = []

        for elem in elements:
            decleration_type = f"{elem.dtype.glsl_type}"
            if elem.dtype.glsl_type_extern is not None:
                decleration_type = f"{elem.dtype.glsl_type_extern}"

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
                if binding.dtype.glsl_type_extern is not None:
                    true_type = binding.dtype.glsl_type_extern

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