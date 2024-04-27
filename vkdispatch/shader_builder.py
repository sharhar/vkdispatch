import vkdispatch as vd

import numpy as np
from enum import Enum

class shader_data_structure(Enum):
    DATA_STRUCTURE_SCALAR = 1,
    DATA_STRUCTURE_VECTOR = 2,
    DATA_STRUCTURE_MATRIX = 3,

class shader_type:
    def __init__(self, name: str, item_size: int, glsl_type: str, structure: shader_data_structure, structure_depth: int, format_str: str, base_type: 'shader_type' = None) -> None:
        self.base_type = base_type if base_type is not None else self
        self.name = name
        self.glsl_type = glsl_type if glsl_type is not None else name
        self.item_size = item_size
        self.structure = structure
        self.structure_depth = structure_depth
        self.format_str = format_str

        assert self.item_size % self.base_type.item_size == 0

        self.item_count = self.item_size // self.base_type.item_size
    
    def __repr__(self) -> str:
        return f"<{self.name}, glsl_type={self.glsl_type} structure_depth={self.structure_depth} item_size={self.item_size} bytes>"

int32 = shader_type("int32", 4, "int", shader_data_structure.DATA_STRUCTURE_SCALAR, 1, "%d")
uint32 = shader_type("uint32", 4, "uint", shader_data_structure.DATA_STRUCTURE_SCALAR, 1, "%u")
float32 = shader_type("float32", 4, "float", shader_data_structure.DATA_STRUCTURE_SCALAR, 1, "%f")
complex64 = shader_type("complex64", 8, "vec2", shader_data_structure.DATA_STRUCTURE_SCALAR, 2, "(%f, %f)")

vec2 = shader_type("vec2", 8, "vec2", shader_data_structure.DATA_STRUCTURE_VECTOR, 2, "(%f, %f)", float32)
vec4 = shader_type("vec4", 16, "vec4", shader_data_structure.DATA_STRUCTURE_VECTOR, 4, "(%f, %f, %f, %f)", float32)

ivec2 = shader_type("ivec2", 8, "ivec2", shader_data_structure.DATA_STRUCTURE_VECTOR, 2, "(%d, %d)", int32)
ivec4 = shader_type("ivec4", 16, "ivec4", shader_data_structure.DATA_STRUCTURE_VECTOR, 4, "(%d, %d, %d, %d)", int32)

uvec2 = shader_type("uvec2", 8, "uvec2", shader_data_structure.DATA_STRUCTURE_VECTOR, 2, "(%u, %u)", uint32)
uvec4 = shader_type("uvec4", 16, "uvec4", shader_data_structure.DATA_STRUCTURE_VECTOR, 4, "(%u, %u, %u, %u)", uint32)

mat2 = shader_type("mat2", 16, "mat2", shader_data_structure.DATA_STRUCTURE_MATRIX, 4, "\n[%f, %f]\n[%f, %f]\n", float32)
mat4 = shader_type("mat4", 64, "mat4", shader_data_structure.DATA_STRUCTURE_MATRIX, 16, "\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n[%f, %f, %f, %f]\n", float32)

def from_numpy_dtype(dtype: type) -> shader_type:
    if dtype == np.int32:
        return int32
    elif dtype == np.uint32:
        return uint32
    elif dtype == np.float32:
        return float32
    elif dtype == np.complex64:
        return complex64
    else:
        raise ValueError(f"Unsupported dtype ({dtype})!")

def to_numpy_dtype(shader_type: shader_type) -> type:
    if shader_type == int32:
        return np.int32
    elif shader_type == uint32:
        return np.uint32
    elif shader_type == float32:
        return np.float32
    elif shader_type == complex64:
        return np.complex64
    else:
        raise ValueError(f"Unsupported shader_type ({shader_type})!")

class shader_variable:
    def __init__(self, name: str, builder: 'shader_builder', var_type: shader_type) -> None:
        self.name = name
        self.builder = builder

        self.var_type = var_type
        self.format = var_type.format_str
    
    def printf_args(self) -> str:
        if self.var_type.structure_depth == 1:
            return self.name

        args_list = []

        for i in range(0, self.var_type.structure_depth):
            args_list.append(f"{self.name}[{i}]")

        return ','.join(args_list)

    def __repr__(self) -> str:
        return self.name
    
    def __lt__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(int32, f"{self} < {other}")

    def __le__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(int32, f"{self} <= {other}")
    
    def __eq__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(int32, f"{self} == {other}")
    
    def __ne__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(int32, f"{self} != {other}")
    
    def __gt__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(int32, f"{self} > {other}")
    
    def __ge__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(int32, f"{self} >= {other}")
    
    def __add__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{self} + {other}")
    
    def __sub__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{self} - {other}")
    
    def __mul__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{self} * {other}")
    
    def __truediv__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{self} / {other}")
    
    #def __floordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    return self.builder.make_var(f"{self} / {other}")
    
    def __mod__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{self} % {other}")
    
    def __pow__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"pow({self}, {other})")
    
    def __neg__(self) -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"-{self}")
    
    #def __pos__(self) -> 'shader_variable':
    #    return self.builder.make_var(f"+{self}")
    
    def __abs__(self) -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"abs({self})")
    
    def __invert__(self) -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"~{self}")
    
    def __lshift__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{self} << {other}")
    
    def __rshift__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{self} >> {other}")
    
    def __and__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{self} & {other}")
    
    def __xor__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{self} ^ {other}")
    
    def __or__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{self} | {other}")
    
    def __radd__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{other} + {self}")
    
    def __rsub__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{other} - {self}")
    
    def __rmul__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{other} * {self}")
    
    def __rtruediv__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{other} / {self}")
    
    #def __rfloordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    return self.builder.make_var(f"{other} / {self}")

    def __rmod__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{other} % {self}")
    
    def __rpow__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"pow({other}, {self})")
    
    def __rand__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{other} & {self}")
    
    def __rxor__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{other} ^ {self}")
    
    def __ror__(self, other: 'shader_variable') -> 'shader_variable':
        return self.builder._make_var(self.var_type, f"{other} | {self}")

    def __iadd__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} += {other};\n")
        return self
    
    def __isub__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} -= {other};\n")
        return self
    
    def __imul__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} *= {other};\n")
        return self
    
    def __itruediv__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} /= {other};\n")
        return self
    
    #def __ifloordiv__(self, other: 'shader_variable') -> 'shader_variable':
    #    self.builder.append_contents(f"{self} /= {other};\n")
    #    return self

    def __imod__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} %= {other};\n")
        return self
    
    def __ipow__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} = pow({self}, {other});\n")
        return self
    
    def __ilshift__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} <<= {other};\n")
        return self
    
    def __irshift__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} >>= {other};\n")
        return self
    
    def __iand__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} &= {other};\n")
        return self
    
    def __ixor__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} ^= {other};\n")
        return self
    
    def __ior__(self, other: 'shader_variable') -> 'shader_variable':
        self.builder.append_contents(f"{self} |= {other};\n")
        return self
    
    def __getitem__(self, index: 'tuple[shader_variable, ...] | shader_variable') -> 'shader_variable':
        if isinstance(index, shader_variable) or isinstance(index, int) or isinstance(index, float):
            return self.builder._make_var(self.var_type.base_type, f"{self}[{index}]")
        else:
            raise ValueError("Unsupported index count!")



    #def __getitem__(self, index: 'shader_variable') -> 'shader_variable':
    #    return self.builder.make_var(self.var_type, f"{self}[{index}]")
    
    def __setitem__(self, index: 'shader_variable', value: 'shader_variable') -> None:
        if f"{self}.data[{index}]" == str(value):
            return
        self.builder.append_contents(f"{self}[{index}] = {value};\n")

class push_constant_buffer:
    def __init__(self, pc_dict: dict) -> None:
        self.pc_dict = pc_dict
        self.ref_dict = {}
        self.pc_list = [0] * len(pc_dict)
        self.var_types: list[shader_type] = [vd.float32] * len(pc_dict)
        self.size = 0

        for key, val in pc_dict.items():
            ii, var_type = val

            dtype = to_numpy_dtype(var_type.base_type)

            self.ref_dict[key] = ii
            self.pc_list[ii] = np.zeros(shape=(var_type.item_count,), dtype=dtype)
            self.var_types[ii] = var_type

            self.size += var_type.item_size
    
    def __setitem__(self, key: str, value: np.ndarray | int | float) -> None:
        if key not in self.ref_dict:
            raise ValueError(f"Invalid push constant '{key}'!")

        ii = self.ref_dict[key]
        end_ind = 1

        if isinstance(value, np.ndarray) or isinstance(value, list) or isinstance(value, tuple):
            end_ind = min(self.var_types[ii].item_count, len(value))

        self.pc_list[ii][:end_ind] = value
    
    def get_bytes(self):
        return b''.join([elem.tobytes() for elem in self.pc_list])

class shader_builder:
    def __init__(self) -> None:
        self.var_count = 0
        self.binding_count = 0
        self.pc_dict = {}
        self.pc_list = []
        self.pc_size = 0
        self.bindings: list[tuple['vd.buffer', int]] = []
        self.global_x = self._make_var(uint32, "gl_GlobalInvocationID.x")
        self.global_y = self._make_var(uint32, "gl_GlobalInvocationID.y")
        self.global_z = self._make_var(uint32, "gl_GlobalInvocationID.z")
        self.contents = ""

        self.header  = "#version 450\n"
        self.header += "#extension GL_ARB_separate_shader_objects : enable\n"
        self.header += "#extension GL_EXT_debug_printf : enable\n"

    def _make_var(self, var_type: shader_type, var_name: str = None) -> shader_variable:
        new_var = f"var{self.var_count}" if var_name is None else var_name
        if var_name is None:
            self.var_count += 1
        return shader_variable(new_var, self, var_type)

    def static_buffer(self, buff: 'vd.buffer', var_name: str = None) -> shader_variable:
        new_var = self.buffer(buff.var_type, var_name)
        self.bindings.append((buff, self.binding_count - 1))
        return new_var
    
    def dynamic_constant(self, var_type: shader_type, var_name: str) -> None:
        #var_name = var_name if var_name is not None else f"pc{len(self.pc_list)}"
        new_var = self._make_var(var_type, f"PC.{var_name}")
        self.pc_list.append((var_name, var_type, f"{var_type.glsl_type} {var_name};"))
        self.pc_size += var_type.item_size
        return new_var

    def buffer(self, var_type: shader_type, var_name: str = None) -> shader_variable:
        buffer_name = f"buf{self.binding_count}" if var_name is None else var_name

        new_var = self._make_var(var_type, f"{buffer_name}.data")
        self.header += f"layout(set = 0, binding = {self.binding_count}) buffer Buffer{self.binding_count} {{ {var_type.glsl_type} data[]; }} {buffer_name};\n"
        self.binding_count += 1
        return new_var
    
    def if_statement(self, arg: shader_variable) -> None:
        self.append_contents(f"if({arg}) {'{'}\n")

    def else_statement(self) -> None:
        self.append_contents("} else {'\n")

    def return_statement(self, arg = None) -> None:
        arg = arg if arg is not None else ""
        self.append_contents(f"return {arg};\n")
    
    def end_if(self) -> None:
        self.append_contents("}\n")

    def print(self, *args: shader_variable | str) -> None:
        args_list = []

        fmt = ""

        for arg in args:
            if isinstance(arg, shader_variable):
                args_list.append(arg.printf_args())

            if isinstance(arg, str):
                fmt += arg
            else:
                fmt += arg.format            


        self.append_contents(f'debugPrintfEXT("{fmt}", {",".join(args_list)});\n')

    #def printf(self, fmt: str, *args: shader_variable) -> None:
    #    args_list = []

    #    for arg in args:
    #        args_list.append(arg.printf_args())

    #    self.append_contents(f'debugPrintfEXT("{fmt}", {",".join(args_list)});\n')

    def append_contents(self, contents: str) -> None:
        self.contents += contents

    def build(self, x: int, y: int, z: int) -> str:
        self.pc_list.sort(key=lambda x: x[1].item_size, reverse=True)
        self.pc_dict = {elem[0]: (ii, elem[1]) for ii, elem in enumerate(self.pc_list)}

        if self.pc_list:
            push_constant_contents = "\n".join([f"\t{elem[2]}" for elem in self.pc_list])

            self.header += f"\nlayout(push_constant) uniform PushConstant {{\n { push_constant_contents } \n}} PC;\n"

        return self.header + f"\nlayout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;\nvoid main() {'{'}\n" + self.contents + "\n}"
    