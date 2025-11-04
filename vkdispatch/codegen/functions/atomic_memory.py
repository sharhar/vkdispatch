import vkdispatch.base.dtype as dtypes
from ..variables.base_variable import BaseVariable
from .arithmetic import is_number
from typing import Any, Union, Tuple

import numpy as np

from .common_builtins import dtype_to_floating, resolve_input


# https://docs.vulkan.org/glsl/latest/chapters/builtinfunctions.html#atomic-memory-functions

def atomic_add(mem: BaseVariable, y: Any) -> BaseVariable:
    raise NotImplementedError("atomic_add is not implemented yet")

    # assert isinstance(mem, BaseVariable), "mem must be a BaseVariable"

    # new_var = self.make_var(arg1.var_type, None, [])
    # self.append_contents(f"{new_var.var_type.glsl_type} {new_var.name} = atomicAdd({arg1.resolve()}, {arg2.resolve()});\n")

    # return mem.new_var(
    #     mem.var_type,
    #     f"atomicAdd({mem.resolve()}, {resolve_input(y)})",
    #     parents=[y, x],
    #     lexical_unit=True
    # )