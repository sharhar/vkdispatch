import vkdispatch.base.dtype as dtypes
from ..variables.base_variable import BaseVariable

from . import utils

def subgroup_add(arg1: BaseVariable):
    return utils.new_var(arg1.var_type, f"subgroupAdd({arg1})", [arg1], lexical_unit=True)

def subgroup_mul(arg1: BaseVariable):
    return utils.new_var(arg1.var_type, f"subgroupMul({arg1})", [arg1], lexical_unit=True)

def subgroup_min(arg1: BaseVariable):
    return utils.new_var(arg1.var_type, f"subgroupMin({arg1})", [arg1], lexical_unit=True)

def subgroup_max(arg1: BaseVariable):
    return utils.new_var(arg1.var_type, f"subgroupMax({arg1})", [arg1], lexical_unit=True)

def subgroup_and(arg1: BaseVariable):
    return utils.new_var(arg1.var_type, f"subgroupAnd({arg1})", [arg1], lexical_unit=True)

def subgroup_or(arg1: BaseVariable):
    return utils.new_var(arg1.var_type, f"subgroupOr({arg1})", [arg1], lexical_unit=True)

def subgroup_xor(arg1: BaseVariable):
    return utils.new_var(arg1.var_type, f"subgroupXor({arg1})", [arg1], lexical_unit=True)

def subgroup_elect():
    return utils.new_var(dtypes.int32, f"subgroupElect()", [], lexical_unit=True)

def subgroup_barrier():
    utils.append_contents("subgroupBarrier();\n")
