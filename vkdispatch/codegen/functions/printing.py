from ..variables.variables import ShaderVariable
from typing import Any
from . import utils

def resolve_arg(arg: Any):
    if isinstance(arg, str):
        return arg
    
    return utils.resolve_input(arg)

def printf(format: str, *args: Any):
    resolved_args = [resolve_arg(arg) for arg in args]
    utils.append_contents(utils.codegen_backend().printf_statement(format, resolved_args) + "\n")

def print_vars(*args: Any, seperator=" "):
    args_list = []

    fmts = []

    for arg in args:
        if isinstance(arg, ShaderVariable):
            args_list.append(arg.printf_args())
            fmts.append(arg.var_type.format_str)
        else:
            fmts.append(str(arg))

    fmt = seperator.join(fmts)
    
    utils.append_contents(utils.codegen_backend().printf_statement(fmt, args_list) + "\n")
