import vkdispatch.base.dtype as dtypes
from vkdispatch.base.dtype import dtype, is_scalar, is_vector, is_matrix, is_complex, to_vector

import vkdispatch.codegen as vc

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

import numpy as np

ENABLE_SCALED_AND_OFFSET_INT = True

def check_is_int(variable):
    return isinstance(variable, int) or np.issubdtype(type(variable), np.integer)

def do_scaled_int_check(other):
    return ENABLE_SCALED_AND_OFFSET_INT and check_is_int(other)

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

class ShaderVariable:
    var_type: dtype
    name: str
    raw_name: str
    can_index: bool = False
    use_child_type: bool = True
    _varying: bool = False
    lexical_unit: bool = False
    settable: bool = False
    parents: List["ShaderVariable"]

    def __init__(self,
                 var_type: dtype, 
                 name: Optional[str] = None,
                 raw_name: Optional[str] = None,
                 lexical_unit: bool = False,
                 settable: bool = False,
                 parents: List["ShaderVariable"] = None
        ) -> None:
        self.var_type = var_type
        self.lexical_unit = lexical_unit

        self.name = name if name is not None else vc.new_name()
        self.raw_name = raw_name if raw_name is not None else self.name

        self.settable = settable

        if parents is None:
            parents = []

        self.parents = []

        for parent_var in parents:
            if isinstance(parent_var, ShaderVariable):
                self.parents.append(parent_var)

        if is_complex(self.var_type):
            self.real = self.new(self.var_type.child_type, f"{self}.x", [self], lexical_unit=True, settable=settable)
            self.imag = self.new(self.var_type.child_type, f"{self}.y", [self], lexical_unit=True, settable=settable)
            self.x = self.real
            self.y = self.imag

            self._register_shape()
        
        if is_vector(self.var_type):
            self.x = self.new(self.var_type.child_type, f"{self}.x", [self], lexical_unit=True, settable=settable)
            
            if self.var_type.child_count >= 2:
                self.y = self.new(self.var_type.child_type, f"{self}.y", [self], lexical_unit=True, settable=settable)

            if self.var_type.child_count >= 3:
                self.z = self.new(self.var_type.child_type, f"{self}.z", [self], lexical_unit=True, settable=settable)

            if self.var_type.child_count == 4:
                self.w = self.new(self.var_type.child_type, f"{self}.w", [self], lexical_unit=True, settable=settable)
            
            self._register_shape()
        
        if is_matrix(self.var_type):
            self._register_shape()

        self._initilized = True

    def __repr__(self) -> str:
        if self.lexical_unit:
            return self.name

        return f"({self.name})"

    def read_callback(self):
        for parent in self.parents:
            parent.read_callback()

    def write_callback(self):
        for parent in self.parents:
            parent.write_callback()

    def new(self, var_type: dtype, name: str, parents: List["ShaderVariable"], lexical_unit: bool = False, settable: bool = False) -> "ShaderVariable":
        return ShaderVariable(var_type, name, lexical_unit=lexical_unit, settable=settable, parents=parents)
       
    def __getitem__(self, index) -> "ShaderVariable":
        if not self.can_index:
            raise ValueError("Unsupported indexing!")
        
        return_type = self.var_type.child_type if self.use_child_type else self.var_type

        if isinstance(index, tuple):
            assert len(index) == 1, "Only single index is supported for tuple indexing!"
            index = index[0]

        if not isinstance(index, ShaderVariable) and not check_is_int(index):
            raise ValueError(f"Unsupported index {index} of type {type(index)}!")
        
        if isinstance(index, ShaderVariable):
            assert dtypes.is_scalar(index.var_type), "Indexing variable must be a scalar!"
            assert dtypes.is_integer_dtype(index.var_type), "Indexing variable must be an integer type!"
        
        return self.new(return_type, f"{self.name}[{shader_var_name(index)}]", [self], settable=self.settable)

    def __setitem__(self, index, value: "ShaderVariable") -> None:
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        if isinstance(index, slice):
            if index.start is None and index.stop is None and index.step is None:
                self.write_callback()

                if isinstance(value, ShaderVariable):
                    value.read_callback()

                vc.append_contents(f"{self.name} = {shader_var_name(value)};\n")
                return
            else:
                raise ValueError("Unsupported slice!")

        if not self.can_index:
            raise ValueError(f"Unsupported indexing {index}!")
        
        if f"{self.name}[{index}]" == str(value):
            return

        self.write_callback()

        if isinstance(index, ShaderVariable):
            index.read_callback()

        if isinstance(value, ShaderVariable):
            value.read_callback()

        vc.append_contents(f"{self.name}[{shader_var_name(index)}] = {shader_var_name(value)};\n")

    def _register_shape(self, shape_var: "ShaderVariable" = None, shape_name: str = None, use_child_type: bool = True):
        self.shape = shape_var
        self.shape_name = shape_name
        self.can_index = True
        self.use_child_type = use_child_type

    def __bool__(self) -> bool:
        raise ValueError(f"Vkdispatch variables cannot be cast to a python boolean")
 
    def new_scaled_and_offset_int(self, var_type: dtype, name: str, parents: List["ShaderVariable"] = None) -> "ScaledAndOfftsetIntVariable":
        return ScaledAndOfftsetIntVariable(var_type, name, parents=parents)

    def copy(self, var_name: str = None):
        """Create a new variable with the same value as the current variable."""
        new_var = self.new(self.var_type, var_name, [], lexical_unit=True, settable=True)

        self.read_callback()

        vc.append_contents(f"{self.var_type.glsl_type} {new_var.name} = {self};\n")
        return new_var

    def cast_to(self, var_type: dtype):
        return self.new(var_type, f"{var_type.glsl_type}({self.name})", [self], lexical_unit=True)

    def printf_args(self) -> str:
        total_count = np.prod(self.var_type.shape)

        if total_count == 1:
            return self.name

        args_list = []

        for i in range(0, total_count):
            args_list.append(f"{self.name}[{i}]")

        return ",".join(args_list)

    def __lt__(self, other):
        return self.new(dtypes.int32, f"{self} < {other}", [self, other])

    def __le__(self, other):
        return self.new(dtypes.int32, f"{self} <= {other}", [self, other])

    def __eq__(self, other):
        return self.new(dtypes.int32, f"{self} == {other}", [self, other])

    def __ne__(self, other):
        return self.new(dtypes.int32, f"{self} != {other}", [self, other])

    def __gt__(self, other):
        return self.new(dtypes.int32, f"{self} > {other}", [self, other])

    def __ge__(self, other):
        return self.new(dtypes.int32, f"{self} >= {other}", [self, other])

    def __add__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}", [self])
            return result.new_from_self(offset=other)

        return self.new(self.var_type, f"{self} + {other}", [self, other])

    def __sub__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}", [self])
            return result.__sub__(other)
        
        return self.new(self.var_type, f"{self} - {other}", [self, other])

    def __mul__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}", [self])
            return result.__mul__(other)

        return_var_type = self.var_type

        if (self.var_type.dimentions == 2
            and other.var_type.dimentions == 1):
            return_var_type = other.var_type

        if(self.var_type == dtypes.int32 or self.var_type == dtypes.uint32):
            if (isinstance(other, int) and is_int_power_of_2(other)):
                if other == 1:
                    return self

                power = int(np.round(np.log2(other)))

                return self.new(self.var_type, f"{self} << {power}", [self])
            elif (isinstance(other, ShaderVariable) and (other.var_type == dtypes.float32)) or (isinstance(other, float) and np.issubdtype(type(other), np.floating)):
                return_var_type = dtypes.float32

        return self.new(return_var_type, f"{self} * {other}", [self, other])

    def __truediv__(self, other):
        if isinstance(other, int) and is_int_power_of_2(other):
            if other == 1:
                return self
            
            if self.var_type != dtypes.int32 and self.var_type != dtypes.uint32:
                return self.new(self.var_type, f"{self} / {other}", [self, other])

            power = int(np.round(np.log2(other)))

            return self.new(self.var_type, f"{self} >> {power}", [self])

        return self.new(self.var_type, f"{self} / {other}", [self, other])

    # def __floordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    return self.builder.make_var(f"{self} / {other}")

    def __mod__(self, other):
        return self.new(self.var_type, f"{self} % {other}", [self, other])

    def __pow__(self, other):
        other_str = str(other)

        if isinstance(other, ShaderVariable):
            other_str = other.name

        return self.new(self.var_type, f"pow({self.name}, {other_str})", [self, other])

    def __neg__(self):
        return self.new(self.var_type, f"-{self}", [self])

    def __abs__(self):
        return self.new(self.var_type, f"abs({self.name})", [self])

    def __invert__(self):
        return self.new(self.var_type, f"~{self}", [self])

    def __lshift__(self, other):
        return self.new(self.var_type, f"{self} << {other}", [self, other])

    def __rshift__(self, other):
        return self.new(self.var_type, f"{self} >> {other}", [self, other])

    def __and__(self, other):
        return self.new(self.var_type, f"{self} & {other}", [self, other])

    def __xor__(self, other):
        return self.new(self.var_type, f"{self} ^ {other}", [self, other])

    def __or__(self, other):
        return self.new(self.var_type, f"({self} | {other}", [self, other])

    def __radd__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}", [self])
            return result.__radd__(other)

        return self.new(self.var_type, f"{other} + {self}", [self, other])

    def __rsub__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}", [self])
            return result.__rsub__(other)

        return self.new(self.var_type, f"{other} - {self}", [self, other])

    def __rmul__(self, other):
        if do_scaled_int_check(other):
            result = self.new_scaled_and_offset_int(self.var_type, f"{self}", [self])
            return result.__rmul__(other)
        
        return_var_type = self.var_type
        
        if(self.var_type == dtypes.int32 or self.var_type == dtypes.uint32):
            if (isinstance(other, int) and is_int_power_of_2(other)):
                if other == 1:
                    return self

                power = int(np.round(np.log2(other)))

                return self.new(self.var_type, f"{self} << {power}", [self])
            elif (isinstance(other, ShaderVariable) and (other.var_type == dtypes.float32)) or (isinstance(other, float) and np.issubdtype(type(other), np.floating)):
                return_var_type = dtypes.float32

        return self.new(return_var_type, f"{other} * {self}", [self, other])

    def __rtruediv__(self, other):
        return self.new(self.var_type, f"{other} / {self}", [self, other])

    # def __rfloordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    return self.builder.make_var(f"{other} / {self}")

    def __rmod__(self, other):
        return self.new(self.var_type, f"{other} % {self}", [self, other])

    def __rpow__(self, other):
        other_str = str(other)

        if isinstance(other, ShaderVariable):
            other_str = other.name

        return self.new(self.var_type, f"pow({other_str}, {self.name})", [self, other])

    def __rand__(self, other):
        return self.new(self.var_type, f"{other} & {self}", [self, other])

    def __rxor__(self, other):
        return self.new(self.var_type, f"{other} ^ {self}", [self, other])

    def __ror__(self, other):
        return self.new(self.var_type, f"{other} | {self}", [self, other])

    def __iadd__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.read_callback()
        self.write_callback()
        
        if isinstance(other, ShaderVariable):
            other.read_callback()

        vc.append_contents(f"{self} += {other};\n")
        return self

    def __isub__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.read_callback()
        self.write_callback()
        
        if isinstance(other, ShaderVariable):
            other.read_callback()

        vc.append_contents(f"{self} -= {other};\n")
        return self

    def __imul__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.read_callback()
        self.write_callback()
        
        if isinstance(other, ShaderVariable):
            other.read_callback()

        vc.append_contents(f"{self} *= {other};\n")
        return self

    def __itruediv__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.read_callback()
        self.write_callback()
        
        if isinstance(other, ShaderVariable):
            other.read_callback()

        vc.append_contents(f"{self} /= {other};\n")
        return self

    # def __ifloordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    self.append_func(f"{self} /= {other};\n")
    #    return self

    def __imod__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.read_callback()
        self.write_callback()
        
        if isinstance(other, ShaderVariable):
            other.read_callback()

        vc.append_contents(f"{self} %= {other};\n")
        return self

    def __ipow__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.read_callback()
        self.write_callback()
        
        other_str = str(other)
        
        if isinstance(other, ShaderVariable):
            other.read_callback()
            other_str = other.name

        vc.append_contents(f"{self} = pow({self.name}, {other_str});\n")
        return self

    def __ilshift__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.read_callback()
        self.write_callback()
        
        if isinstance(other, ShaderVariable):
            other.read_callback()

        vc.append_contents(f"{self} <<= {other};\n")
        return self

    def __irshift__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.read_callback()
        self.write_callback()
        
        if isinstance(other, ShaderVariable):
            other.read_callback()

        vc.append_contents(f"{self} >>= {other};\n")
        return self

    def __iand__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.read_callback()
        self.write_callback()
        
        if isinstance(other, ShaderVariable):
            other.read_callback()

        vc.append_contents(f"{self} &= {other};\n")
        return self

    def __ixor__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.read_callback()
        self.write_callback()
        
        if isinstance(other, ShaderVariable):
            other.read_callback()

        vc.append_contents(f"{self} ^= {other};\n")
        return self

    def __ior__(self, other):
        assert self.settable, f"Cannot set value of '{self.name}' because it is not a settable variable!"

        self.read_callback()
        self.write_callback()
        
        if isinstance(other, ShaderVariable):
            other.read_callback()

        vc.append_contents(f"{self} |= {other};\n")
        return self


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

    def __repr__(self) -> str:
        scale_str = f" * {self.scale}" if self.scale != 1 else ""
        offset_str = f" + {self.offset}" if self.offset != 0 else ""

        if scale_str == "" and offset_str == "":
            return self.base_name

        return f"({self.base_name}{scale_str}{offset_str})"

    def __add__(self, other) -> "Union[ShaderVariable, ScaledAndOfftsetIntVariable]":
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
