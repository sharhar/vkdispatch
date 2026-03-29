import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import Any, Union, Tuple

from . import utils
from . import scalar_eval as se

def comment(comment: str, preceding_new_line: bool = True) -> None:
    comment_text = str(comment).replace("\r\n", "\n").replace("\r", "\n")
    comment_lines = comment_text.split("\n")

    if preceding_new_line:
        utils.append_contents("\n")

    if len(comment_lines) == 1:
        safe_comment = comment_lines[0].replace("*/", "* /")
        utils.append_contents(f"/* {safe_comment} */\n")
        return

    utils.append_contents("/*\n")

    for line in comment_lines:
        safe_line = line.replace("*/", "* /")

        if safe_line:
            utils.append_contents(f" * {safe_line}\n")
            continue

        utils.append_contents(" *\n")

    utils.append_contents(" */\n")

def abs(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return abs(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"abs({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def sign(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.sign(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"sign({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def floor(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.floor(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"floor({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def ceil(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.ceil(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"ceil({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def trunc(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.trunc(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"trunc({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def round(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.round(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"round({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def round_even(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.round(var)
    
    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")
    
    utils.mark_backend_feature("roundEven")

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"roundEven({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def fract(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return float(var - se.floor(var))

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")
    
    utils.mark_backend_feature("fract")

    return utils.new_var(
        utils.dtype_to_floating(var.var_type),
        f"fract({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def mod(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return se.mod(x, y)
    
    base_var = None

    if isinstance(y, ShaderVariable):
        base_var = y
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise ValueError("Arguments must be ShaderVariables or numbers")
    
    utils.mark_backend_feature("mod")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"mod({utils.resolve_input(x)}, {utils.resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def modf(x: Any, y: Any) -> Tuple[ShaderVariable, ShaderVariable]:
    if utils.is_number(y) and utils.is_number(x):
        a, b = se.modf(x, y)
        return float(a), float(b)
    
    if utils.is_number(x) and isinstance(y, ShaderVariable):
        utils.mark_backend_feature("mod")
        return utils.new_var(
            utils.dtype_to_floating(y.var_type),
            f"mod({utils.resolve_input(x)}, {y.resolve()})",
            parents=[y]
        )
    
    if utils.is_number(y) and isinstance(x, ShaderVariable):
        utils.mark_backend_feature("mod")
        return utils.new_var(
            utils.dtype_to_floating(x.var_type),
            f"mod({x.resolve()}, {utils.resolve_input(y)})",
            parents=[x]
        )
    
    if not isinstance(y, ShaderVariable):
        raise ValueError(f"First argument must be a ShaderVariable or number, but got {type(y)}!")
    
    if not isinstance(x, ShaderVariable):
        raise ValueError(f"Second argument must be a ShaderVariable or number, but got {type(x)}!")
    
    utils.mark_backend_feature("mod")

    return utils.new_var(
        utils.dtype_to_floating(y.var_type),
        f"mod({x.resolve()}, {y.resolve()})",
        parents=[y, x],
        lexical_unit=True
    )

def min(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return se.minimum(x, y)
    
    base_var = None

    if isinstance(y, ShaderVariable):
        base_var = y
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise ValueError("Arguments must be ShaderVariables or numbers")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"min({utils.resolve_input(x)}, {utils.resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def max(x: Any, y: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(y) and utils.is_number(x):
        return se.maximum(x, y)
    
    base_var = None

    if isinstance(y, ShaderVariable):
        base_var = y
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise ValueError("Arguments must be ShaderVariables or numbers")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"max({utils.resolve_input(x)}, {utils.resolve_input(y)})",
        parents=[y, x],
        lexical_unit=True
    )

def clip(x: Any, min_val: Any, max_val: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(x) and utils.is_number(min_val) and utils.is_number(max_val):
        return se.clip(x, min_val, max_val)
    
    base_var = None

    if isinstance(min_val, ShaderVariable):
        base_var = min_val
    elif isinstance(max_val, ShaderVariable):
        base_var = max_val
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise ValueError("Arguments must be ShaderVariables or numbers")
    
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
        return se.interp(a, [0, 1], [x, y])
    
    base_var = None

    if isinstance(a, ShaderVariable):
        base_var = a
    elif isinstance(y, ShaderVariable):
        base_var = y
    elif isinstance(x, ShaderVariable):
        base_var = x
    else:
        raise ValueError("Arguments must be ShaderVariables or numbers")
    
    utils.mark_backend_feature("mix")

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
        raise ValueError("Arguments must be ShaderVariables or numbers")
    
    utils.mark_backend_feature("step")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"step({utils.resolve_input(edge)}, {utils.resolve_input(x)})",
        parents=[edge, x],
        lexical_unit=True
    )
    
def smoothstep(edge0: Any, edge1: Any, x: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(edge0) and utils.is_number(edge1) and utils.is_number(x):
        t = se.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return float(t * t * (3.0 - 2.0 * t))
    
    base_var = None

    if isinstance(x, ShaderVariable):
        base_var = x
    elif isinstance(edge1, ShaderVariable):
        base_var = edge1
    elif isinstance(edge0, ShaderVariable):
        base_var = edge0
    else:
        raise ValueError("Arguments must be ShaderVariables or numbers")
    
    utils.mark_backend_feature("smoothstep")

    return utils.new_var(
        utils.dtype_to_floating(base_var.var_type),
        f"smoothstep({utils.resolve_input(edge0)}, {utils.resolve_input(edge1)}, {utils.resolve_input(x)})",
        parents=[edge0, edge1, x],
        lexical_unit=True
    )

def isnan(var: Any) -> Union[ShaderVariable, bool]:
    if utils.is_number(var):
        return se.isnan(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        dtypes.int32,
        f"isnan({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def isinf(var: Any) -> Union[ShaderVariable, bool]:
    if utils.is_number(var):
        return se.isinf(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        dtypes.int32,
        f"isinf({var.resolve()})",
        parents=[var],
        lexical_unit=True
    )

def float_bits_to_int(var: Any) -> Union[ShaderVariable, int]:
    if utils.is_number(var):
        return se.float_bits_to_int(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        dtypes.int32,
        utils.codegen_backend().float_bits_to_int_expr(var.resolve()),
        parents=[var],
        lexical_unit=True
    )

def float_bits_to_uint(var: Any) -> Union[ShaderVariable, int]:
    if utils.is_number(var):
        return se.float_bits_to_uint(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        dtypes.uint32,
        utils.codegen_backend().float_bits_to_uint_expr(var.resolve()),
        parents=[var],
        lexical_unit=True
    )

def int_bits_to_float(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.int_bits_to_float(var)
    
    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        dtypes.float32,
        utils.codegen_backend().int_bits_to_float_expr(var.resolve()),
        parents=[var],
        lexical_unit=True
    )

def uint_bits_to_float(var: Any) -> Union[ShaderVariable, float]:
    if utils.is_number(var):
        return se.uint_bits_to_float(var)

    if not isinstance(var, ShaderVariable):
        raise ValueError(f"Argument must be a ShaderVariable or number, but got {type(var)}!")

    return utils.new_var(
        dtypes.float32,
        utils.codegen_backend().uint_bits_to_float_expr(var.resolve()),
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
        raise ValueError("Arguments must be ShaderVariables or numbers")

    result_type = utils.dtype_to_floating(base_var.var_type)
    fma_function = utils.codegen_backend().fma_function_name(result_type)

    return utils.new_var(
        result_type,
        f"{fma_function}({utils.resolve_input(a)}, {utils.resolve_input(b)}, {utils.resolve_input(c)})",
        parents=[a, b, c],
        lexical_unit=True
    )
