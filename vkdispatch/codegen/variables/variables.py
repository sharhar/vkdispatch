import vkdispatch.base.dtype as dtypes

from ..shader_writer import append_contents, new_name

from .base_variable import BaseVariable

from ..struct_builder import StructElement

from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import Any

import enum
import dataclasses

from ..functions.base_functions import arithmetic
from ..functions.base_functions import bitwise
from ..functions.base_functions import arithmetic_comparisons
from ..functions.base_functions import base_utils

#from ..functions.type_casting import to_dtype
#from ..functions.registers import new_register

ENABLE_SCALED_AND_OFFSET_INT = True

def is_int_power_of_2(n: int) -> bool:
    """Check if an integer is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0

def shader_var_name(index: "Union[Any, ShaderVariable]") -> str:
    if isinstance(index, ShaderVariable):
        return index.resolve()
    
    return str(index)

def var_types_to_floating(var_type: dtypes.dtype) -> dtypes.dtype:
    if var_type == dtypes.int32 or var_type == dtypes.uint32:
        return dtypes.float32

    if var_type == dtypes.ivec2 or var_type == dtypes.uvec2:
        return dtypes.vec2

    if var_type == dtypes.ivec3 or var_type == dtypes.uvec3:
        return dtypes.vec3
    
    if var_type == dtypes.ivec4 or var_type == dtypes.uvec4:
        return dtypes.vec4
    
    return var_type

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
    exec_count_name: str

    def make_source(self, x: int, y: int, z: int) -> str:
        layout_str = f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;"
        return f"{self.header}\n{layout_str}\n{self.body}"
    
    def __repr__(self):
        description_string = ""

        description_string += f"Shader Name: {self.name}\n"
        description_string += f"Push Constant Size: {self.pc_size} bytes\n"
        description_string += f"Push Constant Structure: {self.pc_structure}\n"
        description_string += f"Uniform Structure: {self.uniform_structure}\n"
        description_string += f"Binding Types: {self.binding_type_list}\n"
        description_string += f"Binding Access: {self.binding_access}\n"
        description_string += f"Execution Count Name: {self.exec_count_name}\n"
        description_string += f"Header:\n{self.header}\n"
        description_string += f"Body:\n{self.body}\n"
        return description_string

class ShaderVariable(BaseVariable):
    def __init__(self,
                 var_type: dtypes.dtype, 
                 name: Optional[str] = None,
                 raw_name: Optional[str] = None,
                 lexical_unit: bool = False,
                 settable: bool = False,
                 register: bool = False,
                 parents: List["ShaderVariable"] = None
        ) -> None:
        super().__init__(
            var_type,
            name if name is not None else new_name(),
            raw_name,
            lexical_unit,
            settable,
            register,
            parents
        )

        if dtypes.is_complex(self.var_type):
            self.real = ShaderVariable(self.var_type.child_type, f"{self.resolve()}.x", parents=[self], lexical_unit=True, settable=settable)
            self.imag = ShaderVariable(self.var_type.child_type, f"{self.resolve()}.y", parents=[self], lexical_unit=True, settable=settable)
            self.x = self.real
            self.y = self.imag

            self._register_shape()
        
        if dtypes.is_vector(self.var_type):
            self.x = ShaderVariable(self.var_type.child_type, f"{self.resolve()}.x", parents=[self], lexical_unit=True, settable=settable)
            
            if self.var_type.child_count >= 2:
                self.y = ShaderVariable(self.var_type.child_type, f"{self.resolve()}.y", parents=[self], lexical_unit=True, settable=settable)

            if self.var_type.child_count >= 3:
                self.z = ShaderVariable(self.var_type.child_type, f"{self.resolve()}.z", parents=[self], lexical_unit=True, settable=settable)

            if self.var_type.child_count == 4:
                self.w = ShaderVariable(self.var_type.child_type, f"{self.resolve()}.w", parents=[self], lexical_unit=True, settable=settable)
            
            self._register_shape()
        
        if dtypes.is_matrix(self.var_type):
            self._register_shape()

    def _register_shape(self, shape_var: "BaseVariable" = None, shape_name: str = None, use_child_type: bool = True):
        self.shape = shape_var
        self.shape_name = shape_name
        self.can_index = True
        self.use_child_type = use_child_type
       
    def __getitem__(self, index) -> "ShaderVariable":
        if not self.can_index:
            raise ValueError("Unsupported indexing!")
        
        return_type = self.var_type.child_type if self.use_child_type else self.var_type

        if isinstance(index, tuple):
            assert len(index) == 1, "Only single index is supported for tuple indexing!"
            index = index[0]

        if not isinstance(index, ShaderVariable) and not base_utils.is_int_number(index):
            raise ValueError(f"Unsupported index {index} of type {type(index)}!")
        
        if isinstance(index, ShaderVariable):
            assert dtypes.is_scalar(index.var_type), "Indexing variable must be a scalar!"
            assert dtypes.is_integer_dtype(index.var_type), "Indexing variable must be an integer type!"
        
        return ShaderVariable(return_type, f"{self.resolve()}[{shader_var_name(index)}]", [self], settable=self.settable)

    def __setitem__(self, index, value: "ShaderVariable") -> None:
        assert self.settable, f"Cannot set value of '{self.resolve()}' because it is not a settable variable!"

        if isinstance(index, slice):
            if index.start is None and index.stop is None and index.step is None:
                self.write_callback()

                if isinstance(value, ShaderVariable):
                    value.read_callback()

                append_contents(f"{self.resolve()} = {shader_var_name(value)};\n")
                return
            else:
                raise ValueError("Unsupported slice!")

        if not self.can_index:
            raise ValueError(f"Unsupported indexing {index}!")
        
        if f"{self.resolve()}[{index}]" == str(value):
            return

        self.write_callback()

        if isinstance(index, ShaderVariable):
            index.read_callback()

        if isinstance(value, ShaderVariable):
            value.read_callback()

        append_contents(f"{self.resolve()}[{shader_var_name(index)}] = {shader_var_name(value)};\n")

    def __bool__(self) -> bool:
        raise ValueError(f"Vkdispatch variables cannot be cast to a python boolean")

    def to_register(self, var_name: str = None) -> "ShaderVariable":
        new_var = base_utils.new_base_var(
            self.var_type,
            var_name,
            [],
            lexical_unit=True,
            settable=True,
            register=True
        )

        self.read_callback()
        base_utils.append_contents(f"{new_var.var_type.glsl_type} {new_var.name} = {self.resolve()};\n")
        return new_var

    def to_dtype(self, var_type: dtypes.dtype) -> "ShaderVariable":
        return base_utils.new_base_var(
            var_type,
            f"{var_type.glsl_type}({self.resolve()})", 
            [self],
            lexical_unit=True
        )

    def __lt__(self, other) -> "ShaderVariable": return arithmetic_comparisons.less_than(self, other)
    def __le__(self, other) -> "ShaderVariable": return arithmetic_comparisons.less_or_equal(self, other)
    def __eq__(self, other) -> "ShaderVariable": return arithmetic_comparisons.equal_to(self, other)
    def __ne__(self, other) -> "ShaderVariable": return arithmetic_comparisons.not_equal_to(self, other)
    def __gt__(self, other) -> "ShaderVariable": return arithmetic_comparisons.greater_than(self, other)
    def __ge__(self, other) -> "ShaderVariable": return arithmetic_comparisons.greater_or_equal(self, other)

    def __add__(self, other) -> "ShaderVariable": return arithmetic.add(self, other)
    def __sub__(self, other) -> "ShaderVariable": return arithmetic.sub(self, other)
    def __mul__(self, other) -> "ShaderVariable": return arithmetic.mul(self, other)
    def __truediv__(self, other) -> "ShaderVariable": return arithmetic.truediv(self, other)
    def __floordiv__(self, other) -> 'ShaderVariable': return arithmetic.floordiv(self, other)
    def __mod__(self, other) -> "ShaderVariable": return arithmetic.mod(self, other)
    def __pow__(self, other) -> "ShaderVariable": return arithmetic.pow(self, other)
    def __neg__(self) -> "ShaderVariable": return arithmetic.neg(self)
    def __abs__(self) -> "ShaderVariable": return arithmetic.absolute(self)
    def __invert__(self) -> "ShaderVariable": return bitwise.invert(self)
    def __lshift__(self, other) -> "ShaderVariable": return bitwise.lshift(self, other)
    def __rshift__(self, other) -> "ShaderVariable": return bitwise.rshift(self, other)
    def __and__(self, other) -> "ShaderVariable": return bitwise.and_bits(self, other)
    def __xor__(self, other) -> "ShaderVariable": return bitwise.xor_bits(self, other)
    def __or__(self, other) -> "ShaderVariable": return bitwise.or_bits(self, other)

    def __radd__(self, other) -> "ShaderVariable": return arithmetic.add(self, other)
    def __rsub__(self, other) -> "ShaderVariable": return arithmetic.sub(self, other, reverse=True)
    def __rmul__(self, other) -> "ShaderVariable": return arithmetic.mul(self, other)
    def __rtruediv__(self, other) -> "ShaderVariable": return arithmetic.truediv(self, other, reverse=True)
    def __rfloordiv__(self, other) -> "ShaderVariable": return arithmetic.floordiv(self, other, reverse=True)
    def __rmod__(self, other) -> "ShaderVariable": return arithmetic.mod(self, other, reverse=True)
    def __rpow__(self, other) -> "ShaderVariable": return arithmetic.pow(self, other, reverse=True)
    def __rlshift__(self, other) -> "ShaderVariable": return bitwise.lshift(self, other, reverse=True)
    def __rrshift__(self, other) -> "ShaderVariable": return bitwise.rshift(self, other, reverse=True)
    def __rand__(self, other) -> "ShaderVariable": return bitwise.and_bits(self, other)
    def __rxor__(self, other) -> "ShaderVariable": return bitwise.xor_bits(self, other)
    def __ror__(self, other) -> "ShaderVariable": return bitwise.or_bits(self, other)

    def __iadd__(self, other): return arithmetic.add(self, other, inplace=True)
    def __isub__(self, other): return arithmetic.sub(self, other, inplace=True)
    def __imul__(self, other): return arithmetic.mul(self, other, inplace=True)
    def __itruediv__(self, other): return arithmetic.truediv(self, other, inplace=True)
    def __ifloordiv__(self, other): return arithmetic.floordiv(self, other, inplace=True)
    def __imod__(self, other): return arithmetic.mod(self, other, inplace=True)
    def __ipow__(self, other): return arithmetic.pow(self, other, inplace=True)
    def __ilshift__(self, other) -> "ShaderVariable": return bitwise.lshift(self, other, inplace=True)
    def __irshift__(self, other) -> "ShaderVariable": return bitwise.rshift(self, other, inplace=True)
    def __iand__(self, other) -> "ShaderVariable": return bitwise.and_bits(self, other, inplace=True)
    def __ixor__(self, other) -> "ShaderVariable": return bitwise.xor_bits(self, other, inplace=True)
    def __ior__(self, other) -> "ShaderVariable": return bitwise.or_bits(self, other, inplace=True)

class ScaledAndOfftsetIntVariable(ShaderVariable):
    def __init__(self,
                 var_type: dtypes.dtype, 
                 name: str,
                 scale: int = 1,
                 offset: int = 0,
                 parents: List["ShaderVariable"] = None
        ) -> None:
        self.base_name = str(name)
        self.scale = scale
        self.offset = offset
        
        super().__init__(var_type, name, parents=parents)
    
    def new_from_self(self, scale: int = 1, offset: int = 0):
        child_vartype = self.var_type

        if isinstance(scale, float) or isinstance(offset, float):
            child_vartype = var_types_to_floating(self.var_type)

        return ScaledAndOfftsetIntVariable(
            child_vartype,
            f"{self.name}",
            scale=self.scale * scale,
            offset=offset + self.offset * scale,
            parents=self.parents
        )

    def resolve(self) -> str:        
        scale_str = f" * {self.scale}" if self.scale != 1 else ""
        offset_str = f" + {self.offset}" if self.offset != 0 else ""

        if scale_str == "" and offset_str == "":
            return self.base_name

        return f"({self.base_name}{scale_str}{offset_str})"

    def __add__(self, other) -> "Union[ShaderVariable, ScaledAndOfftsetIntVariable]":
        if is_scalar_number(other):
            return self.new_from_self(offset=other)
        
        return super().__add__(other)

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
