import vkdispatch.base.dtype as dtypes
from . import utils

def inf_f32():
    return utils.new_var(
        dtypes.float32,
        utils.codegen_backend().inf_f32_expr(),
        [],
        lexical_unit=True
    )

def ninf_f32():
    return utils.new_var(
        dtypes.float32,
        utils.codegen_backend().ninf_f32_expr(),
        [],
        lexical_unit=True
    )

def inf_f64():
    return utils.new_var(
        dtypes.float64,
        utils.codegen_backend().inf_f64_expr(),
        [],
        lexical_unit=True
    )

def ninf_f64():
    return utils.new_var(
        dtypes.float64,
        utils.codegen_backend().ninf_f64_expr(),
        [],
        lexical_unit=True
    )

def inf_f16():
    return utils.new_var(
        dtypes.float16,
        utils.codegen_backend().inf_f16_expr(),
        [],
        lexical_unit=True
    )

def ninf_f16():
    return utils.new_var(
        dtypes.float16,
        utils.codegen_backend().ninf_f16_expr(),
        [],
        lexical_unit=True
    )

def global_invocation_id():
    return utils.new_var(
        dtypes.uvec3,
        utils.codegen_backend().global_invocation_id_expr(),
        [],
        lexical_unit=True
    )

def local_invocation_id():
    return utils.new_var(
        dtypes.uvec3,
        utils.codegen_backend().local_invocation_id_expr(),
        [],
        lexical_unit=True
    )

def local_invocation_index():
    return utils.new_var(
        dtypes.uint32,
        utils.codegen_backend().local_invocation_index_expr(),
        [],
        lexical_unit=True
    )

def workgroup_id():
    return utils.new_var(
        dtypes.uvec3,
        utils.codegen_backend().workgroup_id_expr(),
        [],
        lexical_unit=True
    )

def workgroup_size():
    return utils.new_var(
        dtypes.uvec3,
        utils.codegen_backend().workgroup_size_expr(),
        [],
        lexical_unit=True
    )

def num_workgroups():
    return utils.new_var(
        dtypes.uvec3,
        utils.codegen_backend().num_workgroups_expr(),
        [],
        lexical_unit=True
    )

def num_subgroups():
    return utils.new_var(
        dtypes.uint32,
        utils.codegen_backend().num_subgroups_expr(),
        [],
        lexical_unit=True
    )

def subgroup_id():
    return utils.new_var(
        dtypes.uint32,
        utils.codegen_backend().subgroup_id_expr(),
        [],
        lexical_unit=True
    )

def subgroup_size():
    return utils.new_var(
        dtypes.uint32,
        utils.codegen_backend().subgroup_size_expr(),
        [],
        lexical_unit=True
    )

def subgroup_invocation_id():
    return utils.new_var(
        dtypes.uint32,
        utils.codegen_backend().subgroup_invocation_id_expr(),
        [],
        lexical_unit=True
    )
