import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable

from . import utils

def subgroup_add(arg1: ShaderVariable):
    return utils.new_var(arg1.var_type, utils.codegen_backend().subgroup_add_expr(arg1.resolve()), [arg1], lexical_unit=True)

def subgroup_mul(arg1: ShaderVariable):
    return utils.new_var(arg1.var_type, utils.codegen_backend().subgroup_mul_expr(arg1.resolve()), [arg1], lexical_unit=True)

def subgroup_min(arg1: ShaderVariable):
    return utils.new_var(arg1.var_type, utils.codegen_backend().subgroup_min_expr(arg1.resolve()), [arg1], lexical_unit=True)

def subgroup_max(arg1: ShaderVariable):
    return utils.new_var(arg1.var_type, utils.codegen_backend().subgroup_max_expr(arg1.resolve()), [arg1], lexical_unit=True)

def subgroup_and(arg1: ShaderVariable):
    return utils.new_var(arg1.var_type, utils.codegen_backend().subgroup_and_expr(arg1.resolve()), [arg1], lexical_unit=True)

def subgroup_or(arg1: ShaderVariable):
    return utils.new_var(arg1.var_type, utils.codegen_backend().subgroup_or_expr(arg1.resolve()), [arg1], lexical_unit=True)

def subgroup_xor(arg1: ShaderVariable):
    return utils.new_var(arg1.var_type, utils.codegen_backend().subgroup_xor_expr(arg1.resolve()), [arg1], lexical_unit=True)

def subgroup_elect():
    return utils.new_var(dtypes.int32, utils.codegen_backend().subgroup_elect_expr(), [], lexical_unit=True)

def subgroup_barrier():
    utils.append_contents(utils.codegen_backend().subgroup_barrier_statement() + "\n")
