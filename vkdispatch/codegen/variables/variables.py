import vkdispatch.base.dtype as dtypes

from .base_variable import BaseVariable

from ..functions.base_functions import arithmetic
from ..functions.base_functions import bitwise
from ..functions.base_functions import arithmetic_comparisons
from ..functions.base_functions import base_utils

from typing import List, Union, Optional

ENABLE_SCALED_AND_OFFSET_INT = True

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

class ShaderVariable(BaseVariable):
    _initilized: bool
    is_complex: bool
    is_conjugate: Optional[bool]

    def __init__(self,
                 var_type: dtypes.dtype, 
                 name: Optional[str] = None,
                 raw_name: Optional[str] = None,
                 lexical_unit: bool = False,
                 settable: bool = False,
                 register: bool = False,
                 parents: List["ShaderVariable"] = None,
                 is_conjugate: bool = False
        ) -> None:
        super().__setattr__("_initilized", False)

        super().__init__(
            var_type,
            name if name is not None else base_utils.new_name(),
            raw_name,
            lexical_unit,
            settable,
            register,
            parents
        )

        self.is_complex = False
        self.is_conjugate = None

        if dtypes.is_complex(self.var_type):
            self.can_index = True
            self.is_complex = True
            self.is_conjugate = is_conjugate
            
            self.real = self.swizzle("x")
            self.imag = self.swizzle("y")

            if is_conjugate:
                self.imag = -self.imag
            
        elif dtypes.is_vector(self.var_type):
            self.can_index = True

            self.x = self.swizzle("x")
            if self.var_type.child_count >= 2: self.y = self.swizzle("y")
            if self.var_type.child_count >= 3: self.z = self.swizzle("z")
            if self.var_type.child_count == 4: self.w = self.swizzle("w")
        elif dtypes.is_matrix(self.var_type):
            self.can_index = True

        self._initilized = True
       
    def __getitem__(self, index) -> "ShaderVariable":
        assert self.can_index, f"Variable '{self.resolve()}' of type '{self.var_type.name}' cannot be indexed into!"

        return_type = self.var_type.child_type if self.use_child_type else self.var_type

        if isinstance(index, tuple):
            assert len(index) == 1, "Only single index is supported, cannot use multi-dimentional indexing!"
            index = index[0]

        if base_utils.is_int_number(index):
            return ShaderVariable(return_type, f"{self.resolve()}[{index}]", parents=[self], settable=self.settable, lexical_unit=True)
        
        assert isinstance(index, ShaderVariable), f"Index must be a ShaderVariable or int type, not {type(index)}!"
        assert dtypes.is_scalar(index.var_type), "Indexing variable must be a scalar!"
        assert dtypes.is_integer_dtype(index.var_type), "Indexing variable must be an integer type!"
        
        return ShaderVariable(return_type, f"{self.resolve()}[{index.resolve()}]", parents=[self, index], settable=self.settable, lexical_unit=True)

    def swizzle(self, components: str) -> "ShaderVariable":
        assert dtypes.is_vector(self.var_type) or dtypes.is_complex(self.var_type) or dtypes.is_scalar(self.var_type), f"Variable '{self.resolve()}' of type '{self.var_type.name}' does not support swizzling!"
        assert self.use_child_type, f"Variable '{self.resolve()}' does not support swizzling!"

        assert len(components) >= 1 and len(components) <= 4, f"Swizzle must have between 1 and 4 components, got {len(components)}!"

        for c in components:
            assert c in ['x', 'y', 'z', 'w'], f"Invalid swizzle component '{c}'!"

        sample_type = self.var_type if dtypes.is_scalar(self.var_type) else self.var_type.child_type
        return_type = sample_type if len(components) == 1 else dtypes.to_vector(sample_type, len(components))

        if dtypes.is_scalar(self.var_type):
            assert all(c == 'x' for c in components), f"Cannot swizzle scalar variable '{self.resolve()}' with components other than 'x'!"

            return ShaderVariable(
                var_type=return_type,
                name=f"{self.resolve()}.{components}",
                parents=[self],
                lexical_unit=True,
                settable=self.settable,
                register=self.register
            )

        if self.var_type.shape[0] < 4:
            assert 'w' not in components, f"Cannot swizzle variable '{self.resolve()}' of type '{self.var_type.name}' with component 'w'!"

        if self.var_type.shape[0] < 3:
            assert 'z' not in components, f"Cannot swizzle variable '{self.resolve()}' of type '{self.var_type.name}' with component 'z'!"
        
        if self.var_type.shape[0] < 2:
            assert 'y' not in components, f"Cannot swizzle variable '{self.resolve()}' of type '{self.var_type.name}' with component 'y'!"

        return ShaderVariable(
            var_type=return_type,
            name=f"{self.resolve()}.{components}",
            parents=[self],
            lexical_unit=True,
            settable=self.settable,
            register=self.register
        )
    
    def conjugate(self) -> "ShaderVariable":
        assert self.is_complex, f"Variable '{self.resolve()}' of type '{self.var_type.name}' is not a complex variable and cannot be conjugated!"

        return ShaderVariable(
            var_type=self.var_type,
            name=self.name,
            raw_name=self.raw_name,
            lexical_unit=self.lexical_unit,
            settable=False,
            register=False,
            parents=[self],
            is_conjugate=not self.is_conjugate
        )

    def set_value(self, value: "ShaderVariable") -> None:
        assert self.settable, f"Cannot set value of '{self.resolve()}' because it is not a settable variable!"

        self.write_callback()
        self.read_callback()

        if base_utils.is_number(value):
            if self.var_type == dtypes.complex64:
                complex_value = complex(value)
                base_utils.append_contents(f"{self.resolve()} = vec2({complex_value.real}, {complex_value.imag});\n")
                return

            base_utils.append_contents(f"{self.resolve()} = {value};\n")
            return

        assert self.var_type == value.var_type, f"Cannot set variable of type '{self.var_type.name}' to value of type '{value.var_type.name}'!"
        value.read_callback()

        base_utils.append_contents(f"{self.resolve()} = {value.resolve()};\n")

    def __setitem__(self, index, value: "ShaderVariable") -> None:
        assert self.settable, f"Cannot set value of '{self.resolve()}' because it is not a settable variable!"

        if isinstance(index, slice):
            assert index.start is None and index.stop is None and index.step is None, "Only full slice (:) is supported!"
            self.set_value(value)
            return
        
        # ignore if setting variable to itself (happens in some inplace operations)
        if f"{self.resolve()}[{index}]" == str(value):
            return

        self[index].set_value(value)

    def __setattr__(self, name: str, value: "ShaderVariable") -> "ShaderVariable":
        if not self._initilized:
            super().__setattr__(name, value)
            return
        
        if dtypes.is_complex(self.var_type) and (name == "real" or name == "imag"):
            if name == "real":
                self.real.set_value(value)
            else:
                self.imag.set_value(value)
            
            return
        
        if dtypes.is_vector(self.var_type) and (name == "x" or name == "y" or name == "z" or name == "w"):
            if name == "x":
                self.x.set_value(value)
            elif name == "y":
                self.y.set_value(value)
            elif name == "z":
                assert self.var_type.shape[0] >= 3, f"Variable '{self.resolve()}' of type '{self.var_type.name}' does not have 'z' component!"
                self.z.set_value(value)
            elif name == "w":
                assert self.var_type.shape[0] == 4, f"Variable '{self.resolve()}' of type '{self.var_type.name}' does not have 'w' component!"
                self.w.set_value(value)
            return

        super().__setattr__(name, value)

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

    def __iadd__(self, other) -> "ShaderVariable": return arithmetic.add(self, other, inplace=True)
    def __isub__(self, other) -> "ShaderVariable": return arithmetic.sub(self, other, inplace=True)
    def __imul__(self, other) -> "ShaderVariable": return arithmetic.mul(self, other, inplace=True)
    def __itruediv__(self, other) -> "ShaderVariable": return arithmetic.truediv(self, other, inplace=True)
    def __ifloordiv__(self, other) -> "ShaderVariable": return arithmetic.floordiv(self, other, inplace=True)
    def __imod__(self, other) -> "ShaderVariable": return arithmetic.mod(self, other, inplace=True)
    def __ipow__(self, other) -> "ShaderVariable": return arithmetic.pow(self, other, inplace=True)
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
        super().__init__(var_type, name, parents=parents)

        self.base_name = str(name)
        self.scale = scale
        self.offset = offset
        
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
        if base_utils.is_scalar_number(other):
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
