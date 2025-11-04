import vkdispatch.base.dtype as dtypes
from vkdispatch.base.dtype import dtype, is_scalar, is_vector, is_matrix, is_complex, to_vector

import vkdispatch.codegen as vc

from .base_variable import BaseVariable

from ..struct_builder import StructElement, StructBuilder

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import Callable
from typing import Any

import enum
import dataclasses

from ..functions import arithmetic
from ..functions import bitwise
from ..functions import arithmetic_comparisons

import numpy as np

ENABLE_SCALED_AND_OFFSET_INT = True

# from utils import check_is_int

# def do_scaled_int_check(other):
#     return ENABLE_SCALED_AND_OFFSET_INT and check_is_int(other)

def is_int_power_of_2(n: int) -> bool:
    """Check if an integer is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0

def shader_var_name(index: "Union[Any, ShaderVariable]") -> str:
    if isinstance(index, ShaderVariable):
        return index.resolve()
    
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
                 var_type: dtype, 
                 name: Optional[str] = None,
                 raw_name: Optional[str] = None,
                 lexical_unit: bool = False,
                 settable: bool = False,
                 parents: List["ShaderVariable"] = None
        ) -> None:
        super().__init__(var_type, name, raw_name, lexical_unit, settable, parents)

    # Override new_var from BaseVariable
    def new_var(self, var_type: dtype, name: str, parents: List["ShaderVariable"], lexical_unit: bool = False, settable: bool = False) -> "ShaderVariable":
        return ShaderVariable(var_type, name, lexical_unit=lexical_unit, settable=settable, parents=parents)
       
    def __getitem__(self, index) -> "ShaderVariable":
        if not self.can_index:
            raise ValueError("Unsupported indexing!")
        
        return_type = self.var_type.child_type if self.use_child_type else self.var_type

        if isinstance(index, tuple):
            assert len(index) == 1, "Only single index is supported for tuple indexing!"
            index = index[0]

        if not isinstance(index, ShaderVariable) and not arithmetic.is_int_number(index):
            raise ValueError(f"Unsupported index {index} of type {type(index)}!")
        
        if isinstance(index, ShaderVariable):
            assert dtypes.is_scalar(index.var_type), "Indexing variable must be a scalar!"
            assert dtypes.is_integer_dtype(index.var_type), "Indexing variable must be an integer type!"
        
        return self.new_var(return_type, f"{self.resolve()}[{shader_var_name(index)}]", [self], settable=self.settable)

    def __setitem__(self, index, value: "ShaderVariable") -> None:
        assert self.settable, f"Cannot set value of '{self.resolve()}' because it is not a settable variable!"

        if isinstance(index, slice):
            if index.start is None and index.stop is None and index.step is None:
                self.write_callback()

                if isinstance(value, ShaderVariable):
                    value.read_callback()

                vc.append_contents(f"{self.resolve()} = {shader_var_name(value)};\n")
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

        vc.append_contents(f"{self.resolve()}[{shader_var_name(index)}] = {shader_var_name(value)};\n")

    def __bool__(self) -> bool:
        raise ValueError(f"Vkdispatch variables cannot be cast to a python boolean")
 
    def new_scaled_var(self,
                        var_type: dtypes.dtype,
                        name: str,
                        scale: int = 1,
                        offset: int = 0,
                        parents: List["BaseVariable"] = None):
        return ScaledAndOfftsetIntVariable(var_type, name, scale=scale, offset=offset, parents=parents)

    def copy(self, var_name: str = None):
        """Create a new variable with the same value as the current variable."""
        new_var = self.new(self.var_type, var_name, [], lexical_unit=True, settable=True)

        self.read_callback()

        vc.append_contents(f"{self.var_type.glsl_type} {new_var.name} = {self};\n")
        return new_var

    #Override cast_to from BaseVariable, to make return type ShaderVariable
    def cast_to(self, var_type: dtype) -> "ShaderVariable":
        return self.new_var(var_type, f"{var_type.glsl_type}({self.name})", [self], lexical_unit=True)

    def printf_args(self) -> str:
        total_count = np.prod(self.var_type.shape)

        if total_count == 1:
            return self.name

        args_list = []

        for i in range(0, total_count):
            args_list.append(f"{self.name}[{i}]")

        return ",".join(args_list)

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
        if arithmetic.is_scalar_number(other):
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
