from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

import vkdispatch as vd


class ShaderVariable:
    """TODO: Docstring"""

    append_func: Callable[[str], None]
    name_func: Callable[[str], str]
    var_type: vd.dtype
    name: str
    binding: int
    shape: "ShaderVariable"

    def __init__(
        self,
        append_func: Callable[[str], None],
        name_func: Callable[[str], str],
        var_type: vd.dtype,
        name: str = None,
        binding: int = None,
    ) -> None:
        self.append_func = append_func
        self.name_func = name_func
        self.var_type = var_type
        self.name = self.name_func(name)
        self.binding = binding
        self.format = var_type.format_str

        #self._shape = None

        self.can_index = var_type.structure == vd.dtype_structure.DATA_STRUCTURE_BUFFER or var_type.structure == vd.dtype_structure.DATA_STRUCTURE_MATRIX

        if self.var_type.is_complex and self.var_type.structure == vd.dtype_structure.DATA_STRUCTURE_SCALAR:
            self.real = self.new(self.var_type.parent, f"{self}.x")
            self.imag = self.new(self.var_type.parent, f"{self}.y")
            self.x = self.real
            self.y = self.imag
        
        if self.var_type.structure == vd.dtype_structure.DATA_STRUCTURE_VECTOR:
            self.x = self.new(self.var_type.parent, f"{self}.x")
            
            if self.var_type.child_count >= 2:
                self.y = self.new(self.var_type.parent, f"{self}.y")

            if self.var_type.child_count >= 3:
                self.z = self.new(self.var_type.parent, f"{self}.z")

            if self.var_type.child_count == 4:
                self.w = self.new(self.var_type.parent, f"{self}.w")

        self._initilized = True

    def new(self, var_type: vd.dtype, name: str = None):
        return ShaderVariable(self.append_func, self.name_func, var_type, name)

    def copy(self, var_name: str = None):
        """Create a new variable with the same value as the current variable."""
        new_var = self.new(self.var_type, var_name)
        self.append_func(f"{self.var_type.glsl_type} {new_var} = {self};\n")
        return new_var

    def set(self, value: "ShaderVariable") -> None:
        self.append_func(f"{self} = {value};\n")

    def cast_to(self, var_type: vd.dtype):
        return self.new(var_type, f"{var_type.glsl_type}({self})")

    def printf_args(self) -> str:
        if self.var_type.total_count == 1:
            return self.name

        args_list = []

        for i in range(0, self.var_type.total_count):
            args_list.append(f"{self.name}[{i}]")

        return ",".join(args_list)

    def __repr__(self) -> str:
        return self.name

    def __lt__(self, other: "ShaderVariable"):
        return self.new(vd.int32, f"({self} < {other})")

    def __le__(self, other: "ShaderVariable"):
        return self.new(vd.int32, f"({self} <= {other})")

    def __eq__(self, other: "ShaderVariable"):
        return self.new(vd.int32, f"({self} == {other})")

    def __ne__(self, other: "ShaderVariable"):
        return self.new(vd.int32, f"({self} != {other})")

    def __gt__(self, other: "ShaderVariable"):
        return self.new(vd.int32, f"({self} > {other})")

    def __ge__(self, other: "ShaderVariable"):
        return self.new(vd.int32, f"({self} >= {other})")

    def __add__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({self} + {other})")

    def __sub__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({self} - {other})")

    def __mul__(self, other: "ShaderVariable"):
        return_var_type = self.var_type

        if (self.var_type.structure == vd.dtype_structure.DATA_STRUCTURE_MATRIX
            and other.var_type.structure == vd.dtype_structure.DATA_STRUCTURE_VECTOR):
            return_var_type = other.var_type

        return self.new(return_var_type, f"({self} * {other})")

    def __truediv__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({self} / {other})")

    # def __floordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    return self.builder.make_var(f"{self} / {other}")

    def __mod__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({self} % {other})")

    def __pow__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"pow({self}, {other})")

    def __neg__(self):
        return self.new(self.var_type, f"(-{self})")

    def __abs__(self):
        return self.new(self.var_type, f"abs({self})")

    def __invert__(self):
        return self.new(self.var_type, f"(~{self})")

    def __lshift__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({self} << {other})")

    def __rshift__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({self} >> {other})")

    def __and__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({self} & {other})")

    def __xor__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({self} ^ {other})")

    def __or__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({self} | {other})")

    def __radd__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({other} + {self})")

    def __rsub__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({other} - {self})")

    def __rmul__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({other} * {self})")

    def __rtruediv__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({other} / {self})")

    # def __rfloordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    return self.builder.make_var(f"{other} / {self}")

    def __rmod__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({other} % {self})")

    def __rpow__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"pow({other}, {self})")

    def __rand__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({other} & {self})")

    def __rxor__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({other} ^ {self})")

    def __ror__(self, other: "ShaderVariable"):
        return self.new(self.var_type, f"({other} | {self})")

    def __iadd__(self, other: "ShaderVariable"):
        self.append_func(f"{self} += {other};\n")
        return self

    def __isub__(self, other: "ShaderVariable"):
        self.append_func(f"{self} -= {other};\n")
        return self

    def __imul__(self, other: "ShaderVariable"):
        self.append_func(f"{self} *= {other};\n")
        return self

    def __itruediv__(self, other: "ShaderVariable"):
        self.append_func(f"{self} /= {other};\n")
        return self

    # def __ifloordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    self.append_func(f"{self} /= {other};\n")
    #    return self

    def __imod__(self, other: "ShaderVariable"):
        self.append_func(f"{self} %= {other};\n")
        return self

    def __ipow__(self, other: "ShaderVariable"):
        self.append_func(f"{self} = pow({self}, {other});\n")
        return self

    def __ilshift__(self, other: "ShaderVariable"):
        self.append_func(f"{self} <<= {other};\n")
        return self

    def __irshift__(self, other: "ShaderVariable"):
        self.append_func(f"{self} >>= {other};\n")
        return self

    def __iand__(self, other: "ShaderVariable"):
        self.append_func(f"{self} &= {other};\n")
        return self

    def __ixor__(self, other: "ShaderVariable"):
        self.append_func(f"{self} ^= {other};\n")
        return self

    def __ior__(self, other: "ShaderVariable"):
        self.append_func(f"{self} |= {other};\n")
        return self

    def __getitem__(self, index):
        if not self.can_index:
            raise ValueError("Unsupported indexing!")

        if isinstance(index, ShaderVariable) or isinstance(index, (int, np.integer)):
            return self.new(self.var_type.parent, f"{self}[{index}]")
        
        if isinstance(index, tuple):
            if len(index) == 1:
                return self.new(self.var_type.parent, f"{self}[{index[0]}]")
            elif len(index) == 2:
                true_index = f"{index[0]} * {self.shape.y} + {index[1]}"
                return self.new(self.var_type.parent, f"{self}[{true_index}]")
            elif len(index) == 3:
                true_index = f"{index[0]} * {self.shape.y} + {index[1]}"
                true_index = f"({true_index}) * {self.shape.z} + {index[2]}"
                return self.new(self.var_type.parent, f"{self}[{true_index}]")
            else:
                raise ValueError(f"Unsupported number of indicies {len(index)}!")

        #else:
        #    raise ValueError(f"Unsupported index type {index}!")

    def __setitem__(self, index, value: "ShaderVariable") -> None:        
        if isinstance(index, slice):
            if index.start is None and index.stop is None and index.step is None:
                self.append_func(f"{self} = {value};\n")
                return
            else:
                raise ValueError("Unsupported slice!")

        if not self.can_index:
            raise ValueError(f"Unsupported indexing {index}!")
        
        if f"{self}[{index}]" == str(value):
            return

        self.append_func(f"{self}[{index}] = {value};\n")

    def _register_shape(self, shape_var: "ShaderVariable", shape_name: str):
        super().__setattr__("shape", shape_var)
        super().__setattr__("shape_name", shape_name)
    
    def __setattr__(self, name: str, value: "ShaderVariable") -> "ShaderVariable":
        try:
            if self._initilized:
                if self.var_type.structure == vd.dtype_structure.DATA_STRUCTURE_SCALAR and self.var_type.is_complex:
                    if name == "real":
                        self.append_func(f"{self}.x = {value};\n")
                        return
                    
                    if name == "imag":
                        self.append_func(f"{self}.y = {value};\n")
                        return
                
                    if name == "x" or name == "y":
                        self.append_func(f"{self}.{name} = {value};\n")
                        return
                
                if self.var_type.structure == vd.dtype_structure.DATA_STRUCTURE_VECTOR:
                    if name == "x" or name == "y" or name == "z" or name == "w":
                        self.append_func(f"{self}.{name} = {value};\n")
                        return
        except:
            super().__setattr__(name, value)
            return
        
        
        if hasattr(self, name):
            super().__setattr__(name, value)
            return

        raise AttributeError(f"Cannot set attribute '{name}'")
        
