from typing import Callable
from typing import List

import numpy as np

import vkdispatch as vd


class ShaderVariable:
    """TODO: Docstring"""

    append_func: Callable[[str], None]
    name_func: Callable[[str], str]
    var_type: vd.dtype
    name: str
    binding: int

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

    def new(self, var_type: vd.dtype, name: str = None):
        return ShaderVariable(self.append_func, self.name_func, var_type, name)

    def copy(self, var_name: str = None):
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

        if (
            self.var_type.structure == vd.dtype_structure.DATA_STRUCTURE_MATRIX
            and other.var_type.structure == vd.dtype_structure.DATA_STRUCTURE_VECTOR
        ):
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

    def __getitem__(
        self, index: "typing.Union[typing.Tuple[shader_variable, ...], shader_variable]"
    ):
        if isinstance(index, ShaderVariable) or isinstance(index, (int, np.integer)):
            return self.new(self.var_type.parent, f"{self}[{index}]")
        else:
            raise ValueError("Unsupported index type!")

    def __setitem__(self, index, value: "ShaderVariable") -> None:
        if isinstance(index, slice):
            if index.start is None and index.stop is None and index.step is None:
                self.append_func(f"{self} = {value};\n")
                return
            else:
                raise ValueError("Unsupported slice!")

        if f"{self}[{index}]" == str(value):
            return

        self.append_func(f"{self}[{index}] = {value};\n")
