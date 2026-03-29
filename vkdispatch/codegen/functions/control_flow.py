import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import List, Optional, Union
from . import utils

import contextlib

def proc_bool(arg: Union[ShaderVariable, bool]) -> ShaderVariable:
    if isinstance(arg, bool):
        return "true" if arg else "false"
    
    if isinstance(arg, ShaderVariable):
        return arg.resolve()

    raise TypeError(f"Argument of type {type(arg)} cannot be processed as a boolean.")

@contextlib.contextmanager
def if_block(arg: ShaderVariable):
    utils.append_contents(f"if({proc_bool(arg)}) {{\n")
    utils.scope_increment()
    yield
    utils.scope_decrement() 
    utils.append_contents("}\n")

@contextlib.contextmanager
def else_if_block(arg: ShaderVariable):
    utils.append_contents(f"else if({proc_bool(arg)}) {{\n")
    utils.scope_increment()
    yield
    utils.scope_decrement()
    utils.append_contents("}\n")

@contextlib.contextmanager
def else_block():
    utils.append_contents("else {\n")
    utils.scope_increment()
    yield
    utils.scope_decrement()
    utils.append_contents("}\n")

def return_statement(arg=None):
    if arg is None:
        utils.append_contents("return;\n")
        return

    if isinstance(arg, str):
        arg_expr = arg
    elif isinstance(arg, ShaderVariable) or utils.is_number(arg):
        arg_expr = utils.resolve_input(arg)
    else:
        arg_expr = str(arg)

    utils.append_contents(f"return {arg_expr};\n")

@contextlib.contextmanager
def while_block(arg: ShaderVariable):
    utils.append_contents(f"while({proc_bool(arg)}) {{\n")
    utils.scope_increment()
    yield
    utils.scope_decrement() 
    utils.append_contents("}\n")

@contextlib.contextmanager
def scope_block(indent: bool = True, comment: str = None):
    if comment is None:
        utils.append_contents("{\n")
    else:
        utils.append_contents(f"{{ /* {comment} */\n")
    
    if indent:
        utils.scope_increment()
    
    yield
    
    if indent:
        utils.scope_decrement()
        
    utils.append_contents("}\n")

def any(*args: List[ShaderVariable]):
    return utils.new_var(dtypes.int32, f"({' || '.join([str(proc_bool(elem)) for elem in args])})", list(args))

def all(*args: List[ShaderVariable]):
    return utils.new_var(dtypes.int32, f"({' && '.join([str(proc_bool(elem)) for elem in args])})", list(args))
