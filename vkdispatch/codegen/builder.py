import vkdispatch.base.dtype as dtypes
from vkdispatch.base.dtype import dtype

from .struct_builder import StructElement, StructBuilder

from enum import IntFlag, auto

from typing import Dict
from typing import List
from typing import Union
from typing import Optional

import dataclasses

from .variables.variables import BaseVariable, ShaderVariable, var_types_to_floating, SharedBuffer, BindingType, ShaderDescription, ScaledAndOfftsetIntVariable
from .variables.bound_variables import BufferVariable, ImageVariable


@dataclasses.dataclass
class ShaderBinding:
    """
    A dataclass that represents a bound resource in a shader. Either a 
    buffer or an image.

    Attributes:
        dtype (vd.dtype): The dtype of the resource. If 
            the resource is an image, this should be vd.vec4 
            (since all images are sampled with 4 channels in shaders).
        name (str): The name of the resource within the shader code.
        dimension (int): The dimension of the resource. Set to 0 for
            buffers and 1, 2, or 3 for images.
        binding_type (BindingType): The type of the binding. Either
            STORAGE_BUFFER, UNIFORM_BUFFER, or SAMPLER.
    """
    dtype: dtype
    name: str
    dimension: int
    binding_type: BindingType

class ShaderFlags(IntFlag):
    NONE  = 0
    NO_SUBGROUP_OPS = auto()
    NO_PRINTF = auto()
    NO_EXEC_BOUNDS = auto()

class ShaderBuilder:
    var_count: int
    binding_count: int
    binding_read_access: Dict[int, bool]
    binding_write_access: Dict[int, bool]
    binding_list: List[ShaderBinding]
    shared_buffers: List[SharedBuffer]
    scope_num: int
    pc_struct: StructBuilder
    uniform_struct: StructBuilder
    exec_count: Optional[ShaderVariable]
    contents: str
    pre_header: str
    flags: ShaderFlags

    def __init__(self, flags: ShaderFlags = ShaderFlags.NONE, is_apple_device: bool = False) -> None:
        self.flags = flags
        self.is_apple_device = is_apple_device

        self.pre_header = "#version 450\n"
        self.pre_header += "#extension GL_ARB_separate_shader_objects : require\n"
        self.pre_header += "#extension GL_EXT_scalar_block_layout : require\n"

        if not (self.flags & ShaderFlags.NO_SUBGROUP_OPS):
            self.pre_header += "#extension GL_KHR_shader_subgroup_arithmetic : require\n"

        if not (self.flags & ShaderFlags.NO_PRINTF):
            self.pre_header += "#extension GL_EXT_debug_printf : require\n"
        
        self.global_invocation = self.make_var(dtypes.uvec3, "gl_GlobalInvocationID", [], lexical_unit=True)
        self.local_invocation = self.make_var(dtypes.uvec3, "gl_LocalInvocationID", [], lexical_unit=True)
        self.workgroup = self.make_var(dtypes.uvec3, "gl_WorkGroupID", [], lexical_unit=True)
        self.workgroup_size = self.make_var(dtypes.uvec3, "gl_WorkGroupSize", [], lexical_unit=True)
        self.num_workgroups = self.make_var(dtypes.uvec3, "gl_NumWorkGroups", [], lexical_unit=True)

        self.num_subgroups = self.make_var(dtypes.uint32, "gl_NumSubgroups", [], lexical_unit=True)
        self.subgroup_id = self.make_var(dtypes.uint32, "gl_SubgroupID", [], lexical_unit=True)

        self.subgroup_size = self.make_var(dtypes.uint32, "gl_SubgroupSize", [], lexical_unit=True)
        self.subgroup_invocation = self.make_var(dtypes.uint32, "gl_SubgroupInvocationID", [], lexical_unit=True)
        
        self.reset()

    def reset(self) -> None:
        self.var_count = 0
        self.binding_count = 0
        self.pc_struct = StructBuilder()
        self.uniform_struct = StructBuilder()
        self.binding_list = []
        self.binding_read_access = {}
        self.binding_write_access = {}
        self.shared_buffers = []
        self.scope_num = 1
        self.contents = ""
        self.mapping_index: ShaderVariable = None
        self.kernel_index: ShaderVariable = None
        self.mapping_registers: List[ShaderVariable] = None
        
        self.exec_count = self.declare_constant(dtypes.uvec4, var_name="exec_count")
        
        if not (self.flags & ShaderFlags.NO_EXEC_BOUNDS):
            self.if_statement(self.new_var(
                dtypes.int32,
                f"any(lessThanEqual({self.exec_count.resolve()}.xyz, {self.global_invocation.resolve()}.xyz))",
                []
            ))
            self.return_statement()
            self.end()

    def new_var(self,
                var_type: dtype,
                name: str,
                parents: List["ShaderVariable"],
                lexical_unit: bool = False,
                settable: bool = False,
                register: bool = False) -> "ShaderVariable":
        return ShaderVariable(var_type,
                              name,
                              lexical_unit=lexical_unit,
                              settable=settable,
                              register=register,
                              parents=parents)
    
    def new_scaled_var(self,
                        var_type: dtypes.dtype,
                        name: str,
                        scale: int = 1,
                        offset: int = 0,
                        parents: List[BaseVariable] = None):
        return ScaledAndOfftsetIntVariable(var_type,
                                           name,
                                           scale=scale,
                                           offset=offset,
                                           parents=parents)

    def set_mapping_index(self, index: ShaderVariable):
        self.mapping_index = index

    def set_kernel_index(self, index: ShaderVariable):
        self.kernel_index = index

    def set_mapping_registers(self, registers: ShaderVariable):
        self.mapping_registers = list(registers)

    def append_contents(self, contents: str) -> None:
        self.contents += ("    " * self.scope_num) + contents

    def comment(self, comment: str) -> None:
        self.append_contents("\n")
        self.append_contents(f"/* {comment} */\n")

    def new_name(self) -> str:
        new_var = f"var{self.var_count}"
        self.var_count += 1
        return new_var
    
    # def get_name_func(self, prefix: Optional[str] = None, suffix: Optional[str] = None):
    #     my_prefix = [prefix]
    #     my_suffix = [suffix]
    #     def get_name_val(var_name: Union[str, None] = None):
    #         new_var = f"var{self.var_count}" if var_name is None else var_name
    #         raw_name = new_var
            
    #         if var_name is None:
    #             self.var_count += 1

    #         if my_prefix[0] is not None:
    #             new_var = f"{my_prefix[0]}{new_var}"
    #             my_prefix[0] = None
            
    #         if my_suffix[0] is not None:
    #             new_var = f"{new_var}{my_suffix[0]}"
    #             my_suffix[0] = None

    #         return new_var, raw_name
    #     return get_name_val

    def make_var(self,
                 var_type: dtype,
                 var_name: Optional[str],
                 parents: List[ShaderVariable],
                 lexical_unit: bool = False,
                 settable: bool = False) -> ShaderVariable:
        return ShaderVariable(
            var_type,
            var_name,
            lexical_unit=lexical_unit,
            settable=settable,
            parents=parents
        )
    
    def declare_constant(self, var_type: dtype, count: int = 1, var_name: Optional[str] = None):
        if var_name is None:
            var_name = self.new_name()

        new_var = ShaderVariable(
            var_type=var_type,
            name=f"UBO.{var_name}",
            raw_name=var_name,
            lexical_unit=True,
            settable=False,
            parents=[]
        )

        if count > 1:
            new_var.use_child_type = False
            new_var.can_index = True

        self.uniform_struct.register_element(new_var.raw_name, var_type, count)
        return new_var

    def declare_variable(self, var_type: dtype, count: int = 1, var_name: Optional[str] = None):
        if var_name is None:
            var_name = self.new_name()

        new_var = ShaderVariable(
            var_type=var_type,
            name=f"PC.{var_name}",
            raw_name=var_name,
            lexical_unit=True,
            settable=False,
            parents=[]
        )

        new_var._varying = True

        if count > 1:
            new_var.use_child_type = False
            new_var.can_index = True

        self.pc_struct.register_element(new_var.raw_name, var_type, count)
        return new_var
    
    def declare_buffer(self, var_type: dtype, var_name: Optional[str] = None):
        self.binding_count += 1

        buffer_name = f"buf{self.binding_count}" if var_name is None else var_name
        shape_name = f"{buffer_name}_shape"
        
        self.binding_list.append(ShaderBinding(var_type, buffer_name, 0, BindingType.STORAGE_BUFFER))
        self.binding_read_access[self.binding_count] = False
        self.binding_write_access[self.binding_count] = False

        current_binding_count = self.binding_count

        def read_lambda():
            self.binding_read_access[current_binding_count] = True

        def write_lambda():
            self.binding_write_access[current_binding_count] = True
        
        return BufferVariable(
            var_type,
            self.binding_count,
            f"{buffer_name}.data",
            self.declare_constant(dtypes.ivec4, var_name=shape_name),
            shape_name,
            read_lambda=read_lambda,
            write_lambda=write_lambda
        )
    
    def declare_image(self, dimensions: int, var_name: Optional[str] = None):
        self.binding_count += 1

        image_name = f"tex{self.binding_count}" if var_name is None else var_name
        self.binding_list.append(ShaderBinding(dtypes.vec4, image_name, dimensions, BindingType.SAMPLER))
        self.binding_read_access[self.binding_count] = False
        self.binding_write_access[self.binding_count] = False

        def read_lambda():
            self.binding_read_access[self.binding_count] = True

        def write_lambda():
            self.binding_write_access[self.binding_count] = True
        
        return ImageVariable(
            dtypes.vec4,
            self.binding_count,
            dimensions,
            f"{image_name}",
            read_lambda=read_lambda,
            write_lambda=write_lambda
        )
    
    def shared_buffer(self, var_type: dtype, size: int, var_name: Optional[str] = None):
        if var_name is None:
            var_name = self.new_name()
        
        shape_name = f"{var_name}_shape"

        new_var = BufferVariable(
            var_type,
            -1,
            var_name,
            self.declare_constant(dtypes.ivec4, var_name=shape_name),
            shape_name,
            read_lambda=lambda: None,
            write_lambda=lambda: None
        )

        self.shared_buffers.append(SharedBuffer(var_type, size, new_var.name))

        return new_var
    
    def abs(self, arg: ShaderVariable):
        return self.make_var(arg.var_type, f"abs({arg})", [arg], lexical_unit=True)
    
    def acos(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"acos({arg.resolve()})", [arg], lexical_unit=True)

    def acosh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"acosh({arg.resolve()})", [arg], lexical_unit=True)

    def asin(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"asin({arg.resolve()})", [arg], lexical_unit=True)

    def asinh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"asinh({arg.resolve()})", [arg], lexical_unit=True)

    def atan(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"atan({arg.resolve()})", [arg], lexical_unit=True)
    
    def atan2(self, arg1: ShaderVariable, arg2: ShaderVariable):
        # TODO: correctly handle pure float inputs

        floating_arg1 = var_types_to_floating(arg1.var_type)
        floating_arg2 = var_types_to_floating(arg2.var_type)

        assert floating_arg1 == floating_arg2, f"Both arguments to atan2 ({arg1.var_type} and {arg2.var_type}) must be of the same dimentionality"

        return self.make_var(floating_arg1, f"atan({arg1.resolve()}, {arg2.resolve()})", [arg1, arg2], lexical_unit=True)

    def atanh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"atanh({arg.resolve()})", [arg], lexical_unit=True)
    
    def atomic_add(self, arg1: ShaderVariable, arg2: ShaderVariable):
        if not isinstance(arg1, ShaderVariable):
            raise TypeError("First argument to atomic_add must be a ShaderVariable")
        
        arg1.read_callback()
        arg1.write_callback()

        if isinstance(arg2, ShaderVariable):
            arg2.read_callback()

        new_var = self.make_var(arg1.var_type, None, [])
        self.append_contents(f"{new_var.var_type.glsl_type} {new_var.name} = atomicAdd({arg1.resolve()}, {arg2.resolve()});\n")
        return new_var
    
    def barrier(self):
        if self.is_apple_device:
            self.memory_barrier()

        self.append_contents("barrier();\n")
    
    def ceil(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"ceil({arg.resolve()})", [arg], lexical_unit=True)
    
    def clamp(self, arg: ShaderVariable, min_val: ShaderVariable, max_val: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"clamp({arg.resolve()}, {min_val.resolve()}, {max_val.resolve()})", [arg, min_val, max_val], lexical_unit=True)

    def cos(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"cos({arg})", [arg], lexical_unit=True)
    
    def cosh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"cosh({arg})", [arg], lexical_unit=True)
    
    def cross(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(dtypes.vec3, f"cross({arg1}, {arg2})", [arg1, arg2], lexical_unit=True)
    
    def degrees(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"degrees({arg})", [arg], lexical_unit=True)
    
    def determinant(self, arg: ShaderVariable):
        return self.make_var(dtypes.float32, f"determinant({arg})", [arg], lexical_unit=True)
    
    def distance(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(dtypes.float32, f"distance({arg1}, {arg2})", [arg1, arg2], lexical_unit=True)
    
    def dot(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(dtypes.float32, f"dot({arg1}, {arg2})", [arg1, arg2], lexical_unit=True)
    
    def exp(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"exp({arg})", [arg], lexical_unit=True)
    
    def exp2(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"exp2({arg})", [arg], lexical_unit=True)

    def float_bits_to_int(self, arg: ShaderVariable):
        return self.make_var(dtypes.int32, f"floatBitsToInt({arg})", [arg], lexical_unit=True)
    
    def float_bits_to_uint(self, arg: ShaderVariable):
        return self.make_var(dtypes.uint32, f"floatBitsToUint({arg})", [arg], lexical_unit=True)

    def floor(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"floor({arg})", [arg], lexical_unit=True)
    
    def fma(self, arg1: ShaderVariable, arg2: ShaderVariable, arg3: ShaderVariable):
        # TODO: properly handle type conversion and float inputs

        return self.make_var(arg1.var_type, f"fma({arg1}, {arg2}, {arg3})", [arg1, arg2, arg3], lexical_unit=True)
    
    def int_bits_to_float(self, arg: ShaderVariable):
        return self.make_var(dtypes.float32, f"intBitsToFloat({arg})", [arg], lexical_unit=True)

    def inverse(self, arg: ShaderVariable):
        assert arg.var_type.dimentions == 2, f"Cannot apply inverse to non-matrix type {arg.var_type}"

        return self.make_var(arg.var_type, f"inverse({arg})", [arg], lexical_unit=True)
    
    def inverse_sqrt(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"inversesqrt({arg})", [arg], lexical_unit=True)
    
    def isinf(self, arg: ShaderVariable):
        return self.make_var(dtypes.int32, f"any(isinf({arg}))", [arg], lexical_unit=True)
    
    def isnan(self, arg: ShaderVariable):
        return self.make_var(dtypes.int32, f"any(isnan({arg}))", [arg], lexical_unit=True)

    def length(self, arg: ShaderVariable):
        return self.make_var(dtypes.float32, f"length({arg})", [arg], lexical_unit=True)

    def log(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"log({arg})", [arg], lexical_unit=True)

    def log2(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"log2({arg})", [arg], lexical_unit=True)

    def max(self, arg1: ShaderVariable, arg2: ShaderVariable):
        # TODO: properly handle type conversion and float inputs

        return self.make_var(arg1.var_type, f"max({arg1}, {arg2})", [arg1, arg2], lexical_unit=True)

    def memory_barrier(self):
        self.append_contents("memoryBarrier();\n")

    def memory_barrier_shared(self):
        self.append_contents("memoryBarrierShared();\n")

    def min(self, arg1: ShaderVariable, arg2: ShaderVariable):
        # TODO: properly handle type conversion and float inputs

        return self.make_var(arg1.var_type, f"min({arg1}, {arg2})", [arg1, arg2], lexical_unit=True)
    
    def mix(self, arg1: ShaderVariable, arg2: ShaderVariable, arg3: ShaderVariable):
        # TODO: properly handle type conversion and float inputs

        return self.make_var(arg1.var_type, f"mix({arg1}, {arg2}, {arg3})", [arg1, arg2, arg3],  lexical_unit=True)

    def mod(self, arg1: ShaderVariable, arg2: ShaderVariable):
        # TODO: properly handle type conversion and float inputs

        return self.make_var(arg1.var_type, f"mod({arg1}, {arg2})", [arg1, arg2], lexical_unit=True)
    
    def normalize(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"normalize({arg})", [arg], lexical_unit=True)
    
    def pow(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(arg1.var_type, f"pow({arg1}, {arg2})", [arg1, arg2], lexical_unit=True)
    
    def radians(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"radians({arg})", [arg], lexical_unit=True)
    
    def round(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"round({arg})", [arg], lexical_unit=True)
    
    def round_even(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"roundEven({arg})", [arg], lexical_unit=True)

    def sign(self, arg: ShaderVariable):
        return self.make_var(arg.var_type, f"sign({arg})", [arg], lexical_unit=True)

    def sin(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"sin({arg})", [arg], lexical_unit=True)
    
    def sinh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"sinh({arg})", [arg], lexical_unit=True)
    
    def smoothstep(self, arg1: ShaderVariable, arg2: ShaderVariable, arg3: ShaderVariable):
        # TODO: properly handle type conversion and float inputs

        return self.make_var(arg1.var_type, f"smoothstep({arg1}, {arg2}, {arg3})", [arg1, arg2, arg3], lexical_unit=True)

    def sqrt(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"sqrt({arg})", [arg], lexical_unit=True)
    
    def step(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(arg1.var_type, f"step({arg1}, {arg2})", [arg1, arg2], lexical_unit=True)

    def tan(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"tan({arg})", [arg], lexical_unit=True)
    
    def tanh(self, arg: ShaderVariable):
        return self.make_var(var_types_to_floating(arg.var_type), f"tanh({arg})", [arg], lexical_unit=True)
    
    def transpose(self, arg: ShaderVariable):
        return self.make_var(arg.var_type, f"transpose({arg})", [arg], lexical_unit=True)
    
    def trunc(self, arg: ShaderVariable):
        return self.make_var(arg.var_type, f"trunc({arg})", [arg], lexical_unit=True)

    def uint_bits_to_float(self, arg: ShaderVariable):
        return self.make_var(dtypes.float32, f"uintBitsToFloat({arg})", [arg], lexical_unit=True)
    
    def mult_c64(self, arg1: ShaderVariable, arg2: ShaderVariable):
        new_var = self.make_var(
            arg1.var_type,
            f"vec2({arg1}.x * {arg2}.x - {arg1}.y * {arg2}.y, {arg1}.x * {arg2}.y + {arg1}.y * {arg2}.x)",
            [arg1, arg2],
            lexical_unit=True
        )
        return new_var
    
    def mult_c64_by_const(self, arg1: ShaderVariable, number: complex):
        if isinstance(number, ShaderVariable):
            raise ValueError("Cannot multiply complex number by a variable, use mult_c64 instead.")

        new_var = self.make_var(
            arg1.var_type,
            f"vec2({arg1}.x * {number.real} - {arg1}.y * {number.imag}, {arg1}.x * {number.imag} + {arg1}.y * {number.real})",
            [arg1],
            lexical_unit=True
        )
        return new_var
    
    def mult_conj_c64(self, arg1: ShaderVariable, arg2: ShaderVariable):
        new_var = self.make_var(
            arg1.var_type,
            f"vec2({arg1}.x * {arg2}.x + {arg1}.y * {arg2}.y, {arg1}.y * {arg2}.x - {arg1}.x * {arg2}.y)",
            [arg1, arg2],
            lexical_unit=True
        )
        return new_var

    def proc_bool(self, arg: Union[ShaderVariable, bool]) -> ShaderVariable:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        
        if isinstance(arg, ShaderVariable):
            return arg.resolve()

        raise TypeError(f"Argument of type {type(arg)} cannot be processed as a boolean.")

    def if_statement(self, arg: ShaderVariable, command: Optional[str] = None):
        if command is None:
            self.append_contents(f"if({self.proc_bool(arg)}) {'{'}\n")
            self.scope_num += 1
            return
        
        self.append_contents(f"if({self.proc_bool(arg)})\n")
        self.scope_num += 1
        self.append_contents(f"{command}\n")
        self.scope_num -= 1

    def if_any(self, *args: List[ShaderVariable]):
        self.append_contents(f"if({' || '.join([str(self.proc_bool(elem)) for elem in args])}) {'{'}\n")
        self.scope_num += 1

    def if_all(self, *args: List[ShaderVariable]):
        self.append_contents(f"if({' && '.join([str(self.proc_bool(elem)) for elem in args])}) {'{'}\n")
        self.scope_num += 1

    def else_statement(self):
        self.scope_num -= 1
        self.append_contents("} else {\n")
        self.scope_num += 1

    def else_if_statement(self, arg: ShaderVariable):
        self.scope_num -= 1
        self.append_contents(f"}} else if({self.proc_bool(arg)}) {'{'}\n")
        self.scope_num += 1

    def else_if_any(self, *args: List[ShaderVariable]):
        self.scope_num -= 1
        self.append_contents(f"}} else if({' || '.join([str(self.proc_bool(elem)) for elem in args])}) {'{'}\n")
        self.scope_num += 1
    
    def else_if_all(self, *args: List[ShaderVariable]):
        self.scope_num -= 1
        self.append_contents(f"}} else if({' && '.join([str(self.proc_bool(elem)) for elem in args])}) {'{'}\n")
        self.scope_num += 1

    def return_statement(self, arg=None):
        arg = arg if arg is not None else ""
        self.append_contents(f"return {arg};\n")

    def while_statement(self, arg: ShaderVariable):
        self.append_contents(f"while({self.proc_bool(arg)}) {'{'}\n")
        self.scope_num += 1

    def new_scope(self, indent: bool = True, comment: str = None):
        if comment is None:
            self.append_contents("{\n")
        else:
            self.append_contents("{ " + f"/* {comment} */\n")
        
        if indent:
            self.scope_num += 1

    def end(self, indent: bool = True):
        if indent:
            self.scope_num -= 1
            
        self.append_contents("}\n")

    def logical_and(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(dtypes.int32, f"({arg1} && {arg2})", [arg1, arg2])

    def logical_or(self, arg1: ShaderVariable, arg2: ShaderVariable):
        return self.make_var(dtypes.int32, f"({arg1} || {arg2})", [arg1, arg2])

    def subgroup_add(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupAdd({arg1})", [arg1], lexical_unit=True)

    def subgroup_mul(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMul({arg1})", [arg1], lexical_unit=True)

    def subgroup_min(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMin({arg1})", [arg1], lexical_unit=True)

    def subgroup_max(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupMax({arg1})", [arg1], lexical_unit=True)

    def subgroup_and(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupAnd({arg1})", [arg1], lexical_unit=True)

    def subgroup_or(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupOr({arg1})", [arg1], lexical_unit=True)

    def subgroup_xor(self, arg1: ShaderVariable):
        return self.make_var(arg1.var_type, f"subgroupXor({arg1})", [arg1], lexical_unit=True)

    def subgroup_elect(self):
        return self.make_var(dtypes.int32, f"subgroupElect()", [], lexical_unit=True)

    def subgroup_barrier(self):
        self.append_contents("subgroupBarrier();\n")

    def new(self, var_type: dtype, *args, var_name: Optional[str] = None):
        new_var = self.make_var(var_type, var_name, [], lexical_unit=True, settable=True)

        for arg in args:
            if isinstance(arg, ShaderVariable):
                arg.read_callback()

        decleration_suffix = ""
        if len(args) > 0:
            decleration_suffix = f" = {var_type.glsl_type}({', '.join([str(elem) for elem in args])})"

        self.append_contents(f"{new_var.var_type.glsl_type} {new_var.name}{decleration_suffix};\n")

        return new_var

    def printf(self, format: str, *args: Union[ShaderVariable, str], seperator=" "):
        args_string = ""

        for arg in args:
            args_string += f", {arg}"

        self.append_contents(f'debugPrintfEXT("{format}" {args_string});\n')

    def print_vars(self, *args: Union[ShaderVariable, str], seperator=" "):
        args_list = []

        fmts = []

        for arg in args:
            if isinstance(arg, ShaderVariable):
                args_list.append(arg.printf_args())
                fmts.append(arg.var_type.format_str)
            else:
                fmts.append(str(arg))

        fmt = seperator.join(fmts)
        
        args_argument = ""

        if len(args_list) > 0:
            args_argument = f", {','.join(args_list)}"

        self.append_contents(f'debugPrintfEXT("{fmt}"{args_argument});\n')
    
    def complex_from_euler_angle(self, angle: ShaderVariable):
        return self.make_var(dtypes.vec2, f"vec2({self.cos(angle)}, {self.sin(angle)})", [angle])

    def compose_struct_decleration(self, elements: List[StructElement]) -> str:
        declerations = []

        for elem in elements:
            decleration_type = f"{elem.dtype.glsl_type}"

            decleration_suffix = ""
            if elem.count > 1:
                decleration_suffix = f"[{elem.count}]"

            declerations.append(f"\t{decleration_type} {elem.name}{decleration_suffix};")
        
        return "\n".join(declerations)

    def build(self, name: str) -> ShaderDescription:
        header = "" + self.pre_header

        for shared_buffer in self.shared_buffers:
            header += f"shared {shared_buffer.dtype.glsl_type} {shared_buffer.name}[{shared_buffer.size}];\n"

        uniform_elements = self.uniform_struct.build()
        
        uniform_decleration_contents = self.compose_struct_decleration(uniform_elements)
        if len(uniform_decleration_contents) > 0:
            header += f"\nlayout(set = 0, binding = 0) uniform UniformObjectBuffer {{\n { uniform_decleration_contents } \n}} UBO;\n"

        binding_type_list = [BindingType.UNIFORM_BUFFER]
        binding_access = [(True, False)]  # UBO is read-only
        
        for ii, binding in enumerate(self.binding_list):
            if binding.binding_type == BindingType.STORAGE_BUFFER:
                true_type = binding.dtype.glsl_type

                header += f"layout(set = 0, binding = {ii + 1}) buffer Buffer{ii + 1} {{ {true_type} data[]; }} {binding.name};\n"
                binding_type_list.append(binding.binding_type)
                binding_access.append((
                    self.binding_read_access[ii + 1],
                    self.binding_write_access[ii + 1]
                ))
            else:
                header += f"layout(set = 0, binding = {ii + 1}) uniform sampler{binding.dimension}D {binding.name};\n"
                binding_type_list.append(binding.binding_type)
                binding_access.append((
                    self.binding_read_access[ii + 1],
                    self.binding_write_access[ii + 1]
                ))
        
        pc_elements = self.pc_struct.build()

        pc_decleration_contents = self.compose_struct_decleration(pc_elements)
        
        if len(pc_decleration_contents) > 0:
            header += f"\nlayout(push_constant) uniform PushConstant {{\n { pc_decleration_contents } \n}} PC;\n"

        return ShaderDescription(
            header=header,
            body=f"void main() {{\n{self.contents}\n}}\n",
            name=name,
            pc_size=self.pc_struct.size, 
            pc_structure=pc_elements, 
            uniform_structure=uniform_elements, 
            binding_type_list=[binding.value for binding in binding_type_list],
            binding_access=binding_access,
            exec_count_name=self.exec_count.raw_name
        )