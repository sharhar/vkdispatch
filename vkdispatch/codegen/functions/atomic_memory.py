from typing import Any, List

import vkdispatch.base.dtype as dtypes

from ..variables.base_variable import BaseVariable
from ..variables.bound_variables import BufferVariable
from ..variables.variables import ShaderVariable
from . import utils


def _is_buffer_backed_target(var: ShaderVariable) -> bool:
    stack: List[BaseVariable] = [var]
    visited_ids = set()

    while len(stack) > 0:
        current = stack.pop()
        current_id = id(current)
        if current_id in visited_ids:
            continue
        visited_ids.add(current_id)

        if isinstance(current, BufferVariable):
            return True

        stack.extend(current.parents)

    return False


# https://docs.vulkan.org/glsl/latest/chapters/builtinfunctions.html#atomic-memory-functions
def atomic_add(mem: ShaderVariable, y: Any) -> ShaderVariable:
    assert isinstance(mem, ShaderVariable), f"atomic_add target must be a ShaderVariable, got {type(mem)}"
    assert dtypes.is_scalar(mem.var_type), "atomic_add target must be a scalar lvalue"
    assert mem.is_setable(), "atomic_add target must be a writable lvalue"
    assert not mem.is_register(), "atomic_add does not support register/local variables as target"
    assert _is_buffer_backed_target(mem), "atomic_add target must reference a buffer element (e.g., buf[idx])"

    assert mem.var_type in (dtypes.int32, dtypes.uint32), (
        f"atomic_add currently supports only int32/uint32 targets, got '{mem.var_type.name}'"
    )

    parents: List[BaseVariable] = [mem]

    if isinstance(y, ShaderVariable):
        assert dtypes.is_scalar(y.var_type), "atomic_add increment variable must be scalar"
        assert dtypes.is_integer_dtype(y.var_type), (
            f"atomic_add increment variable must be integer-typed, got '{y.var_type.name}'"
        )
        y.read_callback()
        parents.append(y)
        y_expr = utils.backend_constructor(mem.var_type, y)
    elif utils.is_int_number(y):
        y_expr = utils.backend_constructor(mem.var_type, y)
    elif utils.is_number(y):
        raise TypeError(f"atomic_add increment must be an integer scalar, got {y!r}")
    else:
        raise TypeError(f"atomic_add increment must be an integer scalar or ShaderVariable, got {type(y)}")

    mem.read_callback()
    mem.write_callback()

    result_var = utils.new_var(
        mem.var_type,
        None,
        parents=parents,
        lexical_unit=True,
        settable=True,
        register=True
    )

    atomic_expr = utils.codegen_backend().atomic_add_expr(mem.resolve(), y_expr, mem.var_type)
    utils.append_contents(
        f"{utils.backend_type_name(result_var.var_type)} {result_var.name} = {atomic_expr};\n"
    )

    return result_var
