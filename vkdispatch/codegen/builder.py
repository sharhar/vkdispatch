import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

class ShaderBuilder:
    var_count: int
    binding_count: int
    binding_list: List[Tuple[vd.dtype, str, int]]
    shared_buffers: List[Tuple[vd.dtype, int, "vc.BufferVariable"]]
    scope_num: int
    pc_struct: vc.BufferStructure
    uniform_struct: vc.BufferStructure
    exec_count: vc.ShaderVariable
    contents: str
    pre_header: str

    def __init__(self, enable_subgroup_ops: bool = True, enable_atomic_float_ops: bool = True, enable_printf: bool = True) -> None:
        self.reset()

        self.enable_subgroup_ops = enable_subgroup_ops
        self.enable_atomic_float_ops = enable_atomic_float_ops
        self.enable_printf = enable_printf

        self.pre_header = "#version 450\n"
        self.pre_header += "#extension GL_ARB_separate_shader_objects : enable\n"

        if self.enable_subgroup_ops:
            self.pre_header += "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"
        
        if self.enable_atomic_float_ops:
            self.pre_header += "#extension GL_EXT_shader_atomic_float : enable\n"

        if self.enable_printf:
            self.pre_header += "#extension GL_EXT_debug_printf : enable\n"
        
        self.global_invocation = self.make_var(vd.uvec4, "gl_GlobalInvocationID")
        self.local_invocation = self.make_var(vd.uvec4, "gl_LocalInvocationID")
        self.workgroup = self.make_var(vd.uvec4, "gl_WorkGroupID")
        self.workgroup_size = self.make_var(vd.uvec4, "gl_WorkGroupSize")
        self.num_workgroups = self.make_var(vd.uvec4, "gl_NumWorkGroups")

        self.num_subgroups = self.make_var(vd.uint32, "gl_NumSubgroups")
        self.subgroup_id = self.make_var(vd.uint32, "gl_SubgroupID")

        self.subgroup_size = self.make_var(vd.uint32, "gl_SubgroupSize")
        self.subgroup_invocation = self.make_var(vd.uint32, "gl_SubgroupInvocationID")

    def reset(self) -> None:
        self.var_count = 0
        self.binding_count = 0
        self.pc_struct = vc.BufferStructure()
        self.uniform_struct = vc.BufferStructure()
        self.binding_list = []
        self.shared_buffers = []
        self.scope_num = 1
        self.exec_count = None
        self.contents = ""

    def append_contents(self, contents: str) -> None:
        self.contents += ("\t" * self.scope_num) + contents
    
    def get_name_func(self, prefix: str = None, suffix: str = None):
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

    def make_var(self, var_type: vd.dtype, var_name: str = None, prefix: str = None, suffix: str = None):
        return vc.ShaderVariable(self.append_contents, self.get_name_func(prefix, suffix), var_type, var_name)
    
    def declare_constant(self, var_type: vd.dtype, count: int = 1, var_name: str = None):
        decleration_type = f"{var_type.glsl_type}"
        suffix = None
        if var_type.glsl_type_extern is not None:
            decleration_type = f"{var_type.glsl_type_extern}"
            suffix = ".xyz"

        new_var = self.make_var(var_type, var_name, "UBO.", suffix)

        decleration_suffix = ""
        if count > 1:
            new_var.use_child_type = False
            new_var.can_index = True
            decleration_suffix = f"[{count}]"

        self.uniform_struct.register_element(new_var.raw_name, var_type, f"{decleration_type} {new_var.raw_name}{decleration_suffix};", count)
        return new_var

    def declare_variable(self, var_type: vd.dtype, count: int = 1, var_name: str = None):
        decleration_type = f"{var_type.glsl_type}"
        suffix = None
        if var_type.glsl_type_extern is not None:
            decleration_type = f"{var_type.glsl_type_extern}"
            suffix = ".xyz"
        
        new_var = self.make_var(var_type, var_name, "PC.", suffix)
        new_var._varying = True

        decleration_suffix = ""
        if count > 1:
            new_var.use_child_type = False
            new_var.can_index = True
            decleration_suffix = f"[{count}]"

        self.pc_struct.register_element(new_var.raw_name, var_type, f"{decleration_type} {new_var.raw_name}{decleration_suffix};", count)
        return new_var
    
    def declare_buffer(self, var_type: vd.dtype, var_name: str = None):
        self.binding_count += 1

        buffer_name = f"buf{self.binding_count}" if var_name is None else var_name
        shape_name = f"{buffer_name}_shape"
        
        self.binding_list.append((var_type, buffer_name, 0))
        
        return vc.BufferVariable(
            self.append_contents, 
            self.get_name_func(), 
            var_type,
            self.binding_count,
            f"{buffer_name}.data",
            self.declare_constant(vd.ivec4, var_name=shape_name),
            shape_name
        )
    
    def declare_image(self, dimensions: int, var_name: str = None):
        self.binding_count += 1

        image_name = f"tex{self.binding_count}" if var_name is None else var_name
        self.binding_list.append((vd.vec4, image_name, dimensions))
        
        return vc.ImageVariable(
            self.append_contents, 
            self.get_name_func(), 
            vd.vec4,
            self.binding_count,
            dimensions,
            f"{image_name}"
        )
    
    def build(self, x: int, y: int, z: int) -> Tuple[str, int, Dict[str, Tuple[int, vd.dtype]], Dict[str, Tuple[int, vd.dtype]]]:
        header = "" + self.pre_header

        for shared_buffer in self.shared_buffers:
            header += f"shared {shared_buffer[0].glsl_type} {shared_buffer[2]}[{shared_buffer[1]}];\n"

        uniform_decleration_contents, uniform_dict = self.uniform_struct.build()
        
        if len(uniform_decleration_contents) > 0:
            header += f"\nlayout(set = 0, binding = 0) uniform UniformObjectBuffer {{\n { uniform_decleration_contents } \n}} UBO;\n"

        binding_type_list = [3]
        
        for ii, binding in enumerate(self.binding_list):
            if binding[2] == 0:
                true_type = binding[0].glsl_type
                if binding[0].glsl_type_extern is not None:
                    true_type = binding[0].glsl_type_extern

                header += f"layout(set = 0, binding = {ii + 1}) buffer Buffer{ii + 1} {{ {true_type} data[]; }} {binding[1]};\n"
                binding_type_list.append(1)
            else:
                header += f"layout(set = 0, binding = {ii + 1}) uniform sampler{binding[2]}D {binding[1]};\n"
                binding_type_list.append(5)
        
        pc_decleration_contents, pc_dict = self.pc_struct.build()
        
        if len(pc_decleration_contents) > 0:
            header += f"\nlayout(push_constant) uniform PushConstant {{\n { pc_decleration_contents } \n}} PC;\n"
        
        layout_str = f"layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;"

        return f"{header}\n{layout_str}\nvoid main() {{\n{self.contents}\n}}\n", self.pc_struct.my_size, pc_dict, uniform_dict, binding_type_list

    def shared_buffer(self, var_type: vd.dtype, size: int, var_name: str = None):
        buffer_name = self.get_name_func()(var_name)[0]
        shape_name = f"{buffer_name}_shape"

        new_var = vc.BufferVariable(
            self.append_contents, 
            self.get_name_func(), 
            var_type,
            -1,
            buffer_name,
            self.declare_constant(vd.ivec4, var_name=shape_name),
            shape_name
        )

        self.shared_buffers.append((var_type, size, new_var))

        return new_var
    
    def memory_barrier(self):
        self.append_contents("memoryBarrier();\n")

    def memory_barrier_shared(self):
        self.append_contents("memoryBarrierShared();\n")

    def barrier(self):
        self.append_contents("barrier();\n")

    def if_statement(self, arg: vc.ShaderVariable):
        self.append_contents(f"if({arg}) {'{'}\n")
        self.scope_num += 1

    def if_any(self, *args: List[vc.ShaderVariable]):
        self.append_contents(f"if({' || '.join([str(elem) for elem in args])}) {'{'}\n")
        self.scope_num += 1

    def if_all(self, *args: List[vc.ShaderVariable]):
        self.append_contents(f"if({' && '.join([str(elem) for elem in args])}) {'{'}\n")
        self.scope_num += 1

    def else_statement(self):
        self.append_contents("} else {\n")

    def return_statement(self, arg=None):
        arg = arg if arg is not None else ""
        self.append_contents(f"return {arg};\n")

    def while_statement(self, arg: vc.ShaderVariable):
        self.append_contents(f"while({arg}) {'{'}\n")
        self.scope_num += 1

    def length(self, arg: vc.ShaderVariable):
        return self.make_var(vd.float32, f"length({arg})")

    def end(self):
        self.scope_num -= 1
        self.append_contents("}\n")

    def logical_and(self, arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
        return self.make_var(vd.int32, f"({arg1} && {arg2})")

    def logical_or(self, arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
        return self.make_var(vd.int32, f"({arg1} || {arg2})")

    def ceil(self, arg: vc.ShaderVariable):
        return self.make_var(arg.var_type, f"ceil({arg})")

    def floor(self, arg: vc.ShaderVariable):
        return self.make_var(arg.var_type, f"floor({arg})")

    def exp(self, arg: vc.ShaderVariable):
        return self.make_var(arg.var_type, f"exp({arg})")

    def sin(self, arg: vc.ShaderVariable):
        return self.make_var(arg.var_type, f"sin({arg})")

    def cos(self, arg: vc.ShaderVariable):
        return self.make_var(arg.var_type, f"cos({arg})")

    def tan(self, arg: vc.ShaderVariable):
        return self.make_var(arg.var_type, f"tan({arg})")

    def arctan2(self, arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
        return self.make_var(arg1.var_type, f"atan({arg1}, {arg2})")

    def sqrt(self, arg: vc.ShaderVariable):
        return self.make_var(arg.var_type, f"sqrt({arg})")

    def mod(self, arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
        return self.make_var(arg1.var_type, f"mod({arg1}, {arg2})")

    def max(self, arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
        return self.make_var(arg1.var_type, f"max({arg1}, {arg2})")

    def min(self, arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
        return self.make_var(arg1.var_type, f"min({arg1}, {arg2})")

    def atomic_add(self, arg1: vc.ShaderVariable, arg2: vc.ShaderVariable):
        new_var = self.make_var(arg1.var_type)
        self.append_contents(f"{new_var.var_type.glsl_type} {new_var} = atomicAdd({arg1}, {arg2});\n")
        return new_var

    def subgroup_add(self, arg1: vc.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupAdd({arg1})")

    def subgroup_mul(self, arg1: vc.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMul({arg1})")

    def subgroup_min(self, arg1: vc.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMin({arg1})")

    def subgroup_max(self, arg1: vc.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMax({arg1})")

    def subgroup_and(self, arg1: vc.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupAnd({arg1})")

    def subgroup_or(self, arg1: vc.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupOr({arg1})")

    def subgroup_xor(self, arg1: vc.ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupXor({arg1})")

    def subgroup_elect(self):
        return self.make_var(vd.int32, f"subgroupElect()")

    def subgroup_barrier(self):
        self.append_contents("subgroupBarrier();\n")

    def new(self, var_type: vd.dtype, *args, var_name: str = None):
        new_var = self.make_var(var_type, var_name=var_name) #f"float({arg1})")

        decleration_suffix = ""
        if len(args) > 0:
            decleration_suffix = f" = {var_type.glsl_type}({', '.join([str(elem) for elem in args])})"

        self.append_contents(f"{new_var.var_type.glsl_type} {new_var}{decleration_suffix};\n")

        return new_var

    def new_float(self, *args, var_name: str = None):
        return self.new(vd.float32, *args, var_name=var_name)

    def new_int(self, *args, var_name: str = None):
        return self.new(vd.int32, *args, var_name=var_name)

    def new_uint(self, *args, var_name: str = None):
        return self.new(vd.uint32, *args, var_name=var_name)

    def new_vec2(self, *args, var_name: str = None):
        return self.new(vd.vec2, *args, var_name=var_name)

    def new_vec3(self, *args, var_name: str = None):
        return self.new(vd.vec3, *args, var_name=var_name)

    def new_vec4(self, *args, var_name: str = None):
        return self.new(vd.vec4, *args, var_name=var_name)

    def new_uvec2(self, *args, var_name: str = None):
        return self.new(vd.uvec2, *args, var_name=var_name)

    def new_uvec3(self, *args, var_name: str = None):
        return self.new(vd.uvec3, *args, var_name=var_name)

    def new_uvec4(self, *args, var_name: str = None):
        return self.new(vd.uvec4, *args, var_name=var_name)

    def new_ivec2(self, *args, var_name: str = None):
        return self.new(vd.ivec2, *args, var_name=var_name)

    def new_ivec3(self, *args, var_name: str = None):
        return self.new(vd.ivec3, *args, var_name=var_name)

    def new_ivec4(self, *args, var_name: str = None):
        return self.new(vd.ivec4, *args, var_name=var_name)

    def float_bits_to_int(self, arg: vc.ShaderVariable):
        return self.make_var(vd.int32, f"floatBitsToInt({arg})")

    def int_bits_to_float(self, arg: vc.ShaderVariable):
        return self.make_var(vd.float32, f"intBitsToFloat({arg})")

    def printf(self, format: str, *args: Union[vc.ShaderVariable, str], seperator=" "):
        args_string = ""

        for arg in args:
            args_string += f", {arg}"

        self.append_contents(f'debugPrintfEXT("{format}" {args_string});\n')

    def print_vars(self, *args: Union[vc.ShaderVariable, str], seperator=" "):
        args_list = []

        fmts = []

        for arg in args:
            if isinstance(arg, vc.ShaderVariable):
                args_list.append(arg.printf_args())
                fmts.append(arg.var_type.format_str)
            else:
                fmts.append(str(arg))

        fmt = seperator.join(fmts)
        
        args_argument = ""

        if len(args_list) > 0:
            args_argument = f", {','.join(args_list)}"

        self.append_contents(f'debugPrintfEXT("{fmt}"{args_argument});\n')

    def unravel_index(self, index: vc.ShaderVariable, shape: vc.ShaderVariable):
        new_var = self.new_uvec3()

        new_var.x = index % shape.x
        new_var.y = (index / shape.x) % shape.y
        new_var.z = index / (shape.x * shape.y)

        return new_var
