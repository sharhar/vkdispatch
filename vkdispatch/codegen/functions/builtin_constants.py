import vkdispatch.base.dtype as dtypes

from ..variables.base_variable import BaseVariable

from . import utils

def inf_f32():
    return utils.new_var(
        dtypes.float32,
        "uintBitsToFloat(0x7F800000)",
        [],
        lexical_unit=True
    )

def ninf_f32():
    return utils.new_var(
        dtypes.float32,
        "uintBitsToFloat(0xFF800000)",
        [],
        lexical_unit=True
    )

def global_invocation_id():
    return utils.new_var(
        dtypes.uvec3,
        "gl_GlobalInvocationID",
        [],
        lexical_unit=True
    )

def local_invocation_id():
    return utils.new_var(
        dtypes.uvec3,
        "gl_LocalInvocationID",
        [],
        lexical_unit=True
    )

def workgroup_id():
    return utils.new_var(
        dtypes.uvec3,
        "gl_WorkGroupID",
        [],
        lexical_unit=True
    )

def workgroup_size():
    return utils.new_var(
        dtypes.uvec3,
        "gl_WorkGroupSize",
        [],
        lexical_unit=True
    )

def num_workgroups():
    return utils.new_var(
        dtypes.uvec3,
        "gl_NumWorkGroups",
        [],
        lexical_unit=True
    )

def num_subgroups():
    return utils.new_var(
        dtypes.uint32,
        "gl_NumSubgroups",
        [],
        lexical_unit=True
    )

def subgroup_id():
    return utils.new_var(
        dtypes.uint32,
        "gl_SubgroupID",
        [],
        lexical_unit=True
    )

def subgroup_size():
    return utils.new_var(
        dtypes.uint32,
        "gl_SubgroupSize",
        [],
        lexical_unit=True
    )

def subgroup_invocation_id():
    return utils.new_var(
        dtypes.uint32,
        "gl_SubgroupInvocationID",
        [],
        lexical_unit=True
    )
