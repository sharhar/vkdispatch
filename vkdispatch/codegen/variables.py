from typing import Callable
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

import numpy as np

import vkdispatch as vd

class BaseVariable:
    append_func: Callable[[str], None]
    name_func: Callable[[str], str]
    var_type: vd.dtype
    name: str
    raw_name: str
    can_index: bool = False
    use_child_type: bool = True
    index_suffix: str = ""
    _varying: bool = False

    def __init__(
        self,
        append_func: Callable[[str], None],
        name_func: Callable[[str], Tuple[str, str]], 
        var_type: vd.dtype,
        name: str = None,
    ) -> None:
        self.append_func = append_func
        self.name_func = name_func
        self.var_type = var_type

        both_names = self.name_func(name)
        self.name = both_names[0]
        self.raw_name = both_names[1]

        #self._register_shape()
    
    def __repr__(self) -> str:
        return self.name
    
    def new(self, var_type: vd.dtype, name: str = None):
        return BaseVariable(self.append_func, self.name_func, var_type, name)
    
    def __getitem__(self, index) -> "ShaderVariable":
        if not self.can_index:
            raise ValueError("Unsupported indexing!")
        
        return_type = self.var_type.child_type if self.use_child_type else self.var_type

        if isinstance(index, ShaderVariable) or isinstance(index, (int, np.integer)):
            return self.new(return_type, f"{self}[{index}]")
        

        if isinstance(index, tuple):
            if len(index) == 1:
                return self.new(return_type, f"{self}[{index[0]}]{self.index_suffix}")
            elif self.shape is None:
                raise ValueError("Cannot do multidimentional index into object with no shape!")
            
            if len(index) == 2:
                true_index = f"{index[0]} * {self.shape.y} + {index[1]}"
                return self.new(return_type, f"{self}[{true_index}]{self.index_suffix}")
            elif len(index) == 3:
                true_index = f"{index[0]} * {self.shape.y} + {index[1]}"
                true_index = f"({true_index}) * {self.shape.z} + {index[2]}"
                return self.new(return_type, f"{self}[{true_index}]{self.index_suffix}")
            else:
                raise ValueError(f"Unsupported number of indicies {len(index)}!")

        else:
            raise ValueError(f"Unsupported index type {index} of type {type(index)}!")

    def __setitem__(self, index, value: "ShaderVariable") -> None:        
        if isinstance(index, slice):
            if index.start is None and index.stop is None and index.step is None:
                self.append_func(f"{self} = {value};\n")
                return
            else:
                raise ValueError("Unsupported slice!")

        if not self.can_index:
            raise ValueError(f"Unsupported indexing {index}!")
        
        if f"{self}[{index}]{self.index_suffix}" == str(value):
            return

        self.append_func(f"{self}[{index}]{self.index_suffix} = {value};\n")

    def _register_shape(self, shape_var: "ShaderVariable" = None, shape_name: str = None, use_child_type: bool = True):
        self.shape = shape_var
        self.shape_name = shape_name
        self.can_index = True
        self.use_child_type = use_child_type

class ShaderVariable(BaseVariable):
    def __init__(self, 
                 append_func: Callable[[str], None], 
                 name_func: Callable[[str], Tuple[str, str]], 
                 var_type: vd.dtype, 
                 name: Optional[str] = None
        ) -> None:
        super().__init__(append_func, name_func, var_type, name)

        if vd.is_complex(self.var_type):
            self.real = self.new(self.var_type.child_type, f"{self}.x")
            self.imag = self.new(self.var_type.child_type, f"{self}.y")
            self.x = self.real
            self.y = self.imag

            self._register_shape()
        
        if vd.is_vector(self.var_type):
            self.x = self.new(self.var_type.child_type, f"{self}.x")
            
            if self.var_type.child_count >= 2:
                self.y = self.new(self.var_type.child_type, f"{self}.y")

            if self.var_type.child_count >= 3:
                self.z = self.new(self.var_type.child_type, f"{self}.z")

            if self.var_type.child_count == 4:
                self.w = self.new(self.var_type.child_type, f"{self}.w")
            
            self._register_shape()
        
        if vd.is_matrix(self.var_type):
            self._register_shape()

        self._initilized = True

    def new(self, var_type: vd.dtype, name: str = None):
        return ShaderVariable(self.append_func, self.name_func, var_type, name)

    def copy(self, var_name: str = None):
        """Create a new variable with the same value as the current variable."""
        new_var = self.new(self.var_type, var_name)
        self.append_func(f"{self.var_type.glsl_type} {new_var} = {self};\n")
        return new_var

    def cast_to(self, var_type: vd.dtype):
        return self.new(var_type, f"{var_type.glsl_type}({self})")

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
                if vd.is_complex(self.var_type): #.structure == vd.dtype_structure.DATA_STRUCTURE_SCALAR and self.var_type.is_complex:
                    if name == "real":
                        self.append_func(f"{self}.x = {value};\n")
                        return
                    
                    if name == "imag":
                        self.append_func(f"{self}.y = {value};\n")
                        return
                
                    if name == "x" or name == "y":
                        self.append_func(f"{self}.{name} = {value};\n")
                        return
                
                if vd.is_vector(self.var_type):#self.var_type.structure == vd.dtype_structure.DATA_STRUCTURE_VECTOR:
                    if name == "x" or name == "y" or name == "z" or name == "w":
                        self.append_func(f"{self}.{name} = {value};\n")
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
            return self.new(self.var_type.child_type, f"{self}.{name}")
        
        new_type = vd.to_vector(self.var_type.child_type, len(name))
        return self.new(new_type, f"({self}).{name}")

    def __lt__(self, other):
        return self.new(vd.int32, f"({self} < {other})")

    def __le__(self, other):
        return self.new(vd.int32, f"({self} <= {other})")

    def __eq__(self, other):
        return self.new(vd.int32, f"({self} == {other})")

    def __ne__(self, other):
        return self.new(vd.int32, f"({self} != {other})")

    def __gt__(self, other):
        return self.new(vd.int32, f"({self} > {other})")

    def __ge__(self, other):
        return self.new(vd.int32, f"({self} >= {other})")

    def __add__(self, other):
        return self.new(self.var_type, f"({self} + {other})")

    def __sub__(self, other):
        return self.new(self.var_type, f"({self} - {other})")

    def __mul__(self, other):
        return_var_type = self.var_type

        if (self.var_type.dimentions == 2
            and other.var_type.dimentions == 1):
            return_var_type = other.var_type

        return self.new(return_var_type, f"({self} * {other})")

    def __truediv__(self, other):
        return self.new(self.var_type, f"({self} / {other})")

    # def __floordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    return self.builder.make_var(f"{self} / {other}")

    def __mod__(self, other):
        return self.new(self.var_type, f"({self} % {other})")

    def __pow__(self, other):
        return self.new(self.var_type, f"pow({self}, {other})")

    def __neg__(self):
        return self.new(self.var_type, f"(-{self})")

    def __abs__(self):
        return self.new(self.var_type, f"abs({self})")

    def __invert__(self):
        return self.new(self.var_type, f"(~{self})")

    def __lshift__(self, other):
        return self.new(self.var_type, f"({self} << {other})")

    def __rshift__(self, other):
        return self.new(self.var_type, f"({self} >> {other})")

    def __and__(self, other):
        return self.new(self.var_type, f"({self} & {other})")

    def __xor__(self, other):
        return self.new(self.var_type, f"({self} ^ {other})")

    def __or__(self, other):
        return self.new(self.var_type, f"({self} | {other})")

    def __radd__(self, other):
        return self.new(self.var_type, f"({other} + {self})")

    def __rsub__(self, other):
        return self.new(self.var_type, f"({other} - {self})")

    def __rmul__(self, other):
        return self.new(self.var_type, f"({other} * {self})")

    def __rtruediv__(self, other):
        return self.new(self.var_type, f"({other} / {self})")

    # def __rfloordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    return self.builder.make_var(f"{other} / {self}")

    def __rmod__(self, other):
        return self.new(self.var_type, f"({other} % {self})")

    def __rpow__(self, other):
        return self.new(self.var_type, f"pow({other}, {self})")

    def __rand__(self, other):
        return self.new(self.var_type, f"({other} & {self})")

    def __rxor__(self, other):
        return self.new(self.var_type, f"({other} ^ {self})")

    def __ror__(self, other):
        return self.new(self.var_type, f"({other} | {self})")

    def __iadd__(self, other):
        self.append_func(f"{self} += {other};\n")
        return self

    def __isub__(self, other):
        self.append_func(f"{self} -= {other};\n")
        return self

    def __imul__(self, other):
        self.append_func(f"{self} *= {other};\n")
        return self

    def __itruediv__(self, other):
        self.append_func(f"{self} /= {other};\n")
        return self

    # def __ifloordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    self.append_func(f"{self} /= {other};\n")
    #    return self

    def __imod__(self, other):
        self.append_func(f"{self} %= {other};\n")
        return self

    def __ipow__(self, other):
        self.append_func(f"{self} = pow({self}, {other});\n")
        return self

    def __ilshift__(self, other):
        self.append_func(f"{self} <<= {other};\n")
        return self

    def __irshift__(self, other):
        self.append_func(f"{self} >>= {other};\n")
        return self

    def __iand__(self, other):
        self.append_func(f"{self} &= {other};\n")
        return self

    def __ixor__(self, other):
        self.append_func(f"{self} ^= {other};\n")
        return self

    def __ior__(self, other):
        self.append_func(f"{self} |= {other};\n")
        return self

class BoundVariable(ShaderVariable):
    binding: int = -1

    def __init__(self,
                 append_func: Callable[[str], None],
                 name_func: Callable[[str], str],
                 var_type: vd.dtype,
                 binding: int,
                 name: Optional[str] = None,
            ) -> None:
            super().__init__(append_func, name_func, var_type, name)

            self.binding = binding
    
    def __int__(self):
        return int(self.binding)

class BufferVariable(BoundVariable):
    def __init__(self,
                 append_func: Callable[[str], None],
                 name_func: Callable[[str], Tuple[str, str]], 
                 var_type: vd.dtype,
                 binding: int,
                 name: Optional[str] = None,
                 shape_var: "ShaderVariable" = None,
                 shape_name: Optional[str] = None,
                 raw_name: Optional[str] = None
            ) -> None:
            super().__init__(append_func, name_func, var_type, binding, name)

            self.name = name if name is not None else self.name
            self.raw_name = raw_name if raw_name is not None else self.raw_name

            self._register_shape(shape_var=shape_var, shape_name=shape_name, use_child_type=False)

class ImageVariable(BoundVariable):
    dimensions: int = 0

    def __init__(self,
                 append_func: Callable[[str], None],
                 name_func: Callable[[str], Tuple[str, str]], 
                 var_type: vd.dtype,
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
            return self.new(vd.vec4, f"texture({self}, {sample_coord_string})")
        
        return self.new(vd.vec4, f"textureLod({self}, {sample_coord_string}, {lod})")
    