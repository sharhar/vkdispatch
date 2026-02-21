from ..variables.variables import ShaderVariable
from typing import Any
from . import utils

def resolve_arg(arg: Any):
    if isinstance(arg, str):
        return arg
    
    return utils.resolve_input(arg)

def printf(format: str, *args: Any):
    args_string = ""

    for arg in args:
        args_string += f", {resolve_arg(arg)}"

    utils.append_contents(f'debugPrintfEXT("{format}" {args_string});\n')

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
    
    args_argument = ""

    if len(args_list) > 0:
        args_argument = f", {','.join(args_list)}"

    utils.append_contents(f'debugPrintfEXT("{fmt}"{args_argument});\n')
