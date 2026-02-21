import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Any, Union, Tuple
import numpy as np

from . import utils

def comment(comment: str) -> None:
    utils.append_contents("\n")
    utils.append_contents(f"/* {comment} */\n")

def abs(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return abs(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"abs({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def sign(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.sign(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"sign({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def floor(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.floor(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"floor({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def ceil(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.ceil(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"ceil({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def trunc(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.trunc(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"trunc({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def round(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.round(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"round({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def round_even(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.round(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"roundEven({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def fract(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(var - np.floor(var))

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"fract({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def mod(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return float(np.mod(x, y))
    
    base_var = None

    if isinstance(y, ShaderVariable):
        base_var = y
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"mod({utils.resolve_input(x)}, {utils.resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def modf(x: Any, y: Any) -> Tuple[ShaderVariable, ShaderVariable]:
    if utils.is_number(y) and utils.is_number(x):
        a, b = np.modf(x, y)
        return float(a), float(b)
    
    if utils.is_number(x) and isinstance(y, ShaderVariable):
        return utils.new_var(
            utils.dtype_to_floating(y.var_type),
            f"mod({x}, {y.resolve()})",
            parents=[y]
        )
    
    if utils.is_number(y) and isinstance(x, ShaderVariable):
        return utils.new_var(
            utils.dtype_to_floating(x.var_type),
            f"mod({x.resolve()}, {y})",
            parents=[x]
        )

    assert isinstance(y, ShaderVariable), "First argument must be a ShaderVariable or number"
    assert isinstance(x, ShaderVariable), "Second argument must be a ShaderVariable or number"

    return utils.new_var(
        utils.dtype_to_floating(y.var_type),
        f"mod({x.resolve()}, {y.resolve()})",
        parents=[y, x],
        lexical_unit=True
    )

def min(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return float(np.minimum(x, y))
    
    base_var = None

    if isinstance(y, ShaderVariable):
        base_var = y
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"min({utils.resolve_input(x)}, {utils.resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def max(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return float(np.maximum(x, y))
    
    base_var = None

    if isinstance(y, ShaderVariable):
        base_var = y
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"max({utils.resolve_input(x)}, {utils.resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def clip(x: Any, min_val: Any, max_val: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(x) and utils.is_number(min_val) and utils.is_number(max_val):
        return float(np.clip(x, min_val, max_val))
    
    base_var = None

    if isinstance(min_val, ShaderVariable):
        base_var = min_val
    elif isinstance(max_val, ShaderVariable):
        base_var = max_val
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")
    
    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"clamp({utils.resolve_input(x)}, {utils.resolve_input(min_val)}, {utils.resolve_input(max_val)})",
        parents=[x, min_val, max_val],
        lexical_unit=True
    )

def clamp(x: Any, min_val: Any, max_val: Any) -> Union[ShaderVariable, float]:
    return clip(x, min_val, max_val)

def mix(x: Any, y: Any, a: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x) and utils.is_number(a):
        return float(np.interp(a, [0, 1], [x, y]))
    
    base_var = None

    if isinstance(a, ShaderVariable):
        base_var = a
    elif isinstance(y, ShaderVariable):
        base_var = y
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"mix({utils.resolve_input(x)}, {utils.resolve_input(y)}, {utils.resolve_input(a)})",
        parents=[y, x, a],
        lexical_unit=True
    )

def step(edge: Any, x: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(edge) and utils.is_number(x):
        return float(0.0 if x < edge else 1.0)
    
    base_var = None

    if isinstance(x, ShaderVariable):
        base_var = x
    elif isinstance(edge, ShaderVariable):
        base_var = edge
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"step({utils.resolve_input(edge)}, {utils.resolve_input(x)})",
        parents=[edge, x],
        lexical_unit=True
    )
    
def smoothstep(edge0: Any, edge1: Any, x: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(edge0) and utils.is_number(edge1) and utils.is_number(x):
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return float(t * t * (3.0 - 2.0 * t))
    
    base_var = None

    if isinstance(x, ShaderVariable):
        base_var = x
    elif isinstance(edge1, ShaderVariable):
        base_var = edge1
    elif isinstance(edge0, ShaderVariable):
        base_var = edge0
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"smoothstep({utils.resolve_input(edge0)}, {utils.resolve_input(edge1)}, {utils.resolve_input(x)})",
        parents=[edge0, edge1, x],
        lexical_unit=True
    )

def isnan(var: Any) -> Union[ShaderVariable, bool]:
    if utils.is_number(var):
        return np.isnan(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtypes.int32,
        f"isnan({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def isinf(var: Any) -> Union[ShaderVariable, bool]:
    if utils.is_number(var):
        return np.isinf(var)

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtypes.int32,
        f"isinf({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def float_bits_to_int(var: Any) -> Union[ShaderVariable, int]:
    if utils.is_number(var):
        return int(np.frombuffer(np.float32(var).tobytes(), dtype=np.int32)[0])

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtypes.int32,
        f"floatBitsToInt({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def float_bits_to_uint(var: Any) -> Union[ShaderVariable, int]:
    if utils.is_number(var):
        return int(np.frombuffer(np.float32(var).tobytes(), dtype=np.uint32)[0])

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtypes.uint32,
        f"floatBitsToUint({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def int_bits_to_float(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.frombuffer(np.int32(var).tobytes(), dtype=np.float32)[0])

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtypes.float32,
        f"intBitsToFloat({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def uint_bits_to_float(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(np.frombuffer(np.uint32(var).tobytes(), dtype=np.float32)[0])

    assert isinstance(var, ShaderVariable), "Argument must be a ShaderVariable or number"

    return utils.new_var(
        dtypes.float32,
        f"uintBitsToFloat({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def fma(a: Any, b: Any, c: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(a) and utils.is_number(b) and utils.is_number(c):
        return float(a * b + c)

    base_var = None

    if isinstance(c, ShaderVariable):
        base_var = c
    elif isinstance(b, ShaderVariable):
        base_var = b
    elif isinstance(a, ShaderVariable):
        base_var = a
    else:
        raise AssertionError("Arguments must be ShaderVariables or numbers")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"fma({utils.resolve_input(a)}, {utils.resolve_input(b)}, {utils.resolve_input(c)})",
        parents=[a, b, c],
        lexical_unit=True
    )