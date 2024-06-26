import copy
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

import vkdispatch as vd


class BufferStructureProxy:
    """TODO: Docstring"""

    pc_dict: Dict[str, Tuple[int, vd.dtype]]
    ref_dict: Dict[str, int]
    pc_list: List[np.ndarray]
    var_types: List[vd.dtype]
    numpy_dtypes: List
    size: int
    data_size: int
    prologue: bytes
    index: int
    alignment: int

    def __init__(self, pc_dict: dict, alignment: int) -> None:
        self.pc_dict = copy.deepcopy(pc_dict)
        self.ref_dict = {}
        self.pc_list = [None] * len(self.pc_dict)
        self.var_types = [None] * len(self.pc_dict)
        self.numpy_dtypes = [None] * len(self.pc_dict)
        self.data_size = 0
        self.prologue = b""
        self.size = 0
        self.index = 0
        self.alignment = alignment

        #uniform_alignment = vd.get_context().device_infos[0].uniform_buffer_alignment
        #self.static_constants_size = int(np.ceil(uniform_buffer.size / float(uniform_alignment))) * int(uniform_alignment)

        # Populate the push constant buffer with the given dictionary
        for key, val in self.pc_dict.items():
            ii, var_type = val

            self.ref_dict[key] = ii

            dtype = vd.to_numpy_dtype(var_type.scalar)
            self.numpy_dtypes[ii] = dtype
            self.pc_list[ii] = np.zeros(
                shape=var_type.numpy_shape, dtype=self.numpy_dtypes[ii]
            )
            self.var_types[ii] = var_type

            self.data_size += var_type.item_size
        
        if self.alignment == 0:
            self.size = self.data_size
        else:
            self.size = int(np.ceil(self.data_size / self.alignment)) * self.alignment

            self.prologue = b"\x00" * (self.size - self.data_size)

    def __setitem__(
        self, key: str, value: Union[np.ndarray, list, tuple, int, float]
    ) -> None:
        if key not in self.ref_dict:
            raise ValueError(f"Invalid push constant '{key}'!")

        ii = self.ref_dict[key]

        if (
            not isinstance(value, np.ndarray)
            and not isinstance(value, list)
            and not isinstance(value, tuple)
        ):
            self.pc_list[ii][0] = value
            return

        arr = np.array(value, dtype=self.numpy_dtypes[ii])

        if arr.shape != self.var_types[ii].numpy_shape:
            raise ValueError(
                f"The shape of {key} is {self.var_types[ii].numpy_shape} but {arr.shape} was given!"
            )

        self.pc_list[ii] = arr

    def __repr__(self) -> str:
        result = "Push Constant Buffer:\n"

        for key, val in self.pc_dict.items():
            ii, var_type = val
            result += f"\t{key} ({var_type.name}): {self.pc_list[ii]}\n"

        return result[:-1]

    def get_bytes(self):
        return b"".join([elem.tobytes() for elem in self.pc_list]) + self.prologue


class BufferStructure:
    my_dict: Dict[str, Tuple[int, vd.dtype]]
    my_list: List[Tuple[str, vd.dtype, str]]
    my_size: int

    def __init__(self, ) -> None:
        self.my_dict = {}
        self.my_list = []
        self.my_size = 0
    
    def register_element(self, var_name: str, var_type: vd.dtype, var_decleration: str):
        self.my_list.append((var_name, var_type, var_decleration))
        self.my_size += var_type.item_size

    def build(self) -> Tuple[str, Dict[str, Tuple[int, vd.dtype]]]:
        self.my_list.sort(key=lambda x: x[1].alignment_size, reverse=True)
        self.my_dict = {elem[0]: (ii, elem[1]) for ii, elem in enumerate(self.my_list)}

        if len(self.my_list) == 0:
            return "", None

        buffer_decleration_contents = "\n".join(
            [f"\t{elem[2]}" for elem in self.my_list]
        )

        return buffer_decleration_contents, self.my_dict


class ShaderBuilder:
    """TODO: Docstring"""

    var_count: int
    binding_count: int
    binding_list: List[Tuple[str, str]]
    shared_buffers: List[Tuple[vd.dtype, int, vd.ShaderVariable]]
    scope_num: int
    pc_struct: BufferStructure
    uniform_struct: BufferStructure

    global_x: vd.ShaderVariable
    global_y: vd.ShaderVariable
    global_z: vd.ShaderVariable

    local_x: vd.ShaderVariable
    local_y: vd.ShaderVariable
    local_z: vd.ShaderVariable

    workgroup_x: vd.ShaderVariable
    workgroup_y: vd.ShaderVariable
    workgroup_z: vd.ShaderVariable

    workgroup_size_x: vd.ShaderVariable
    workgroup_size_y: vd.ShaderVariable
    workgroup_size_z: vd.ShaderVariable

    num_workgroups_x: vd.ShaderVariable
    num_workgroups_y: vd.ShaderVariable
    num_workgroups_z: vd.ShaderVariable

    num_subgroups: vd.ShaderVariable
    subgroup_id: vd.ShaderVariable

    subgroup_size: vd.ShaderVariable
    subgroup_invocation: vd.ShaderVariable

    contents: str
    pre_header: str

    def __init__(self) -> None:
        self.var_count = 0
        self.binding_count = 1
        self.pc_struct = BufferStructure()
        self.uniform_struct = BufferStructure()
        self.binding_list = []
        self.shared_buffers = []
        self.scope_num = 1

        self.global_x = self.make_var(vd.uint32, "gl_GlobalInvocationID.x")
        self.global_y = self.make_var(vd.uint32, "gl_GlobalInvocationID.y")
        self.global_z = self.make_var(vd.uint32, "gl_GlobalInvocationID.z")

        self.local_x = self.make_var(vd.uint32, "gl_LocalInvocationID.x")
        self.local_y = self.make_var(vd.uint32, "gl_LocalInvocationID.y")
        self.local_z = self.make_var(vd.uint32, "gl_LocalInvocationID.z")

        self.workgroup_x = self.make_var(vd.uint32, "gl_WorkGroupID.x")
        self.workgroup_y = self.make_var(vd.uint32, "gl_WorkGroupID.y")
        self.workgroup_z = self.make_var(vd.uint32, "gl_WorkGroupID.z")

        self.workgroup_size_x = self.make_var(vd.uint32, "gl_WorkGroupSize.x")
        self.workgroup_size_y = self.make_var(vd.uint32, "gl_WorkGroupSize.y")
        self.workgroup_size_z = self.make_var(vd.uint32, "gl_WorkGroupSize.z")

        self.num_workgroups_x = self.make_var(vd.uint32, "gl_NumWorkGroups.x")
        self.num_workgroups_y = self.make_var(vd.uint32, "gl_NumWorkGroups.y")
        self.num_workgroups_z = self.make_var(vd.uint32, "gl_NumWorkGroups.z")

        self.num_subgroups = self.make_var(vd.uint32, "gl_NumSubgroups")
        self.subgroup_id = self.make_var(vd.uint32, "gl_SubgroupID")

        self.subgroup_size = self.make_var(vd.uint32, "gl_SubgroupSize")
        self.subgroup_invocation = self.make_var(vd.uint32, "gl_SubgroupInvocationID")

        self.contents = ""

        self.pre_header = "#version 450\n"
        self.pre_header += "#extension GL_ARB_separate_shader_objects : enable\n"
        self.pre_header += "#extension GL_EXT_debug_printf : enable\n"
        self.pre_header += "#extension GL_EXT_shader_atomic_float : enable\n"
        self.pre_header += "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"

    def reset(self) -> None:
        self.var_count = 0
        self.binding_count = 1
        self.pc_struct = BufferStructure()
        self.uniform_struct = BufferStructure()
        self.binding_list = []
        self.shared_buffers = []
        self.scope_num = 1
        self.contents = ""

    def get_name(self, var_name: str = None) -> str:
        new_var = f"var{self.var_count}" if var_name is None else var_name
        if var_name is None:
            self.var_count += 1
        return new_var

    def make_var(self, var_type: vd.dtype, var_name: str = None):
        return vd.ShaderVariable(self.append_contents, self.get_name, var_type, var_name)

    def get_variable_decleration(self, var_name: str, var_type: vd.dtype, args: list):
        is_buffer = ""
        suffix = ""

        if args:
            if var_type.structure == vd.dtype_structure.DATA_STRUCTURE_BUFFER:
                raise ValueError("Cannot initialize buffer in variable decleration!")

            suffix = f" = {var_type.glsl_type}({', '.join([str(elem) for elem in args])})"
        
        if var_type.structure == vd.dtype_structure.DATA_STRUCTURE_BUFFER:
            if var_type.child_count == 0:
                is_buffer = "[]"
            else:
                is_buffer = f"[{var_type.child_count}]"

        return f"{var_type.glsl_type} {var_name}{is_buffer}{suffix};"

    def static_constant(self, var_type: vd.dtype, var_name: str):
        new_var = self.make_var(var_type, f"UBO.{var_name}")
        self.uniform_struct.register_element(var_name, var_type, self.get_variable_decleration(var_name, var_type, []))
        return new_var

    def dynamic_constant(self, var_type: vd.dtype, var_name: str):
        new_var = self.make_var(var_type, f"PC.{var_name}")
        self.pc_struct.register_element(var_name, var_type, self.get_variable_decleration(var_name, var_type, []))
        return new_var
    
    def new(self, var_type: vd.dtype, *args, var_name: str = None):
        new_var = self.make_var(var_type, var_name)
        self.append_contents(self.get_variable_decleration(new_var, var_type, args))
        return new_var
    
    def push_constant(self, var_type: vd.dtype, var_name: str):
        return self.dynamic_constant(var_type, var_name)

    def dynamic_buffer(self, var_type: vd.dtype, var_name: str = None):
        buffer_name = f"buf{self.binding_count}" if var_name is None else var_name
        new_var = self.make_var(var_type, f"{buffer_name}.data")
        self.binding_list.append((var_type, buffer_name))
        new_var.binding = self.binding_count
        shape_name = f"{buffer_name}_shape"
        new_var._register_shape(self.static_constant(vd.ivec4, shape_name), shape_name)
        self.binding_count += 1
        return new_var

    def shared_buffer(self, var_type: vd.dtype, size: int, var_name: str = None):
        new_var = self.make_var(var_type[size], var_name)
        self.shared_buffers.append((new_var.var_type, size, new_var))
        return new_var

    def memory_barrier_shared(self):
        self.append_contents("memoryBarrierShared();\n")

    def barrier(self):
        self.append_contents("barrier();\n")

    def if_statement(self, arg: vd.ShaderVariable):
        self.append_contents(f"if({arg}) {'{'}\n")
        self.scope_num += 1

    def if_any(self, *args: List[vd.ShaderVariable]):
        self.append_contents(f"if({' || '.join([str(elem) for elem in args])}) {'{'}\n")
        self.scope_num += 1

    def if_all(self, *args: List[vd.ShaderVariable]):
        self.append_contents(f"if({' && '.join([str(elem) for elem in args])}) {'{'}\n")
        self.scope_num += 1

    def else_statement(self):
        self.append_contents("} else {'\n")
    
    def logical_and(self, arg1: vd.ShaderVariable, arg2: vd.ShaderVariable):
        return self.make_var(vd.int32, f"({arg1} && {arg2})")

    def logical_or(self, arg1: vd.ShaderVariable, arg2: vd.ShaderVariable):
        return self.make_var(vd.int32, f"({arg1} || {arg2})")

    def return_statement(self, arg=None):
        arg = arg if arg is not None else ""
        self.append_contents(f"return {arg};\n")

    def while_statement(self, arg: vd.ShaderVariable):
        self.append_contents(f"while({arg}) {'{'}\n")
        self.scope_num += 1

    def end(self):
        self.scope_num -= 1
        self.append_contents("}\n")

    def ceil(self, arg: vd.ShaderVariable):
        return self.make_var(arg.var_type, f"ceil({arg})")

    def exp(self, arg: vd.ShaderVariable):
        return self.make_var(arg.var_type, f"exp({arg})")

    def sin(self, arg: vd.ShaderVariable):
        return self.make_var(arg.var_type, f"sin({arg})")

    def cos(self, arg: vd.ShaderVariable):
        return self.make_var(arg.var_type, f"cos({arg})")

    def sqrt(self, arg: vd.ShaderVariable):
        return self.make_var(arg.var_type, f"sqrt({arg})")

    def max(self, arg1: vd.ShaderVariable, arg2: vd.ShaderVariable):
        return self.make_var(arg1.var_type, f"max({arg1}, {arg2})")
    
    def min(self, arg1: vd.ShaderVariable, arg2: vd.ShaderVariable):
        return self.make_var(arg1.var_type, f"min({arg1}, {arg2})")

    def atomic_add(self, arg1: vd.ShaderVariable, arg2: vd.ShaderVariable):
        new_var = self.new(arg1.var_type)
        self.append_contents(f"{new_var} = atomicAdd({arg1}, {arg2});\n")
        return new_var

    def subgroup_add(self, arg1: vd.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupAdd({arg1})")
    
    def subgroup_mul(self, arg1: vd.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMul({arg1})")

    def subgroup_min(self, arg1: vd.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMin({arg1})")

    def subgroup_max(self, arg1: vd.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMax({arg1})")
    
    def subgroup_and(self, arg1: vd.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupAnd({arg1})")
    
    def subgroup_or(self, arg1: vd.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupOr({arg1})")

    def subgroup_xor(self, arg1: vd.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupXor({arg1})")

    def subgroup_elect(self):
        return self.make_var(vd.int32, f"subgroupElect()")

    def subgroup_barrier(self):
        self.append_contents("subgroupBarrier();\n")

    def float_bits_to_int(self, arg: vd.ShaderVariable):
        return self.make_var(vd.int32, f"floatBitsToInt({arg})")

    def int_bits_to_float(self, arg: vd.ShaderVariable):
        return self.make_var(vd.float32, f"intBitsToFloat({arg})")

    def print(self, *args: Union[vd.ShaderVariable, str], seperator=" "):
        args_list = []

        fmts = []

        for arg in args:
            if isinstance(arg, vd.ShaderVariable):
                args_list.append(arg.printf_args())
                fmts.append(arg.format)
            else:
                fmts.append(str(arg))

        fmt = seperator.join(fmts)
        
        args_argument = ""

        if len(args_list) > 0:
            args_argument = f", {','.join(args_list)}"

        self.append_contents(f'debugPrintfEXT("{fmt}"{args_argument});\n')

    def append_contents(self, contents: str) -> None:
        self.contents += ("\t" * self.scope_num) + contents

    def build(self, x: int, y: int, z: int) -> Tuple[str, int, Dict[str, Tuple[int, vd.dtype]], Dict[str, Tuple[int, vd.dtype]]]:
        header = "" + self.pre_header

        for shared_buffer in self.shared_buffers:
            header += f"shared {shared_buffer[0].glsl_type} {shared_buffer[2]}[{shared_buffer[1]}];\n"

        uniform_decleration_contents, uniform_dict = self.uniform_struct.build()
        
        if len(uniform_decleration_contents) > 0:
            header += f"\nlayout(set = 0, binding = 0) uniform UniformObjectBuffer {{\n { uniform_decleration_contents } \n}} UBO;\n"
        
        for ii, binding in enumerate(self.binding_list):
            header += f"layout(set = 0, binding = {ii + 1}) buffer Buffer{ii + 1} {{ {self.get_variable_decleration('data', binding[0], [])} }} {binding[1]};\n"
        
        pc_decleration_contents, pc_dict = self.pc_struct.build()
        
        if len(pc_decleration_contents) > 0:
            header += f"\nlayout(push_constant) uniform PushConstant {{\n { pc_decleration_contents } \n}} PC;\n"
        
        layout_str = f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;"

        return f"{header}\n{layout_str}\nvoid main() {{\n{self.contents}\n}}\n", self.pc_struct.my_size, pc_dict, uniform_dict


shader = ShaderBuilder()
