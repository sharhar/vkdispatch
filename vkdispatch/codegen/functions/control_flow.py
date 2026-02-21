import vkdispatch.base.dtype as dtypes
from ..variables.variables import ShaderVariable
from typing import List, Optional, Union
from . import utils

def proc_bool(arg: Union[ShaderVariable, bool]) -> ShaderVariable:
    if isinstance(arg, bool):
        return "true" if arg else "false"
    
    if isinstance(arg, ShaderVariable):
        return arg.resolve()

    raise TypeError(f"Argument of type {type(arg)} cannot be processed as a boolean.")

def if_statement(arg: ShaderVariable, command: Optional[str] = None):
    if command is None:
        utils.append_contents(f"if({proc_bool(arg)}) {'{'}\n")
        utils.scope_increment()
        return
    
    utils.append_contents(f"if({proc_bool(arg)})\n")
    utils.scope_increment()
    utils.append_contents(f"{command}\n")
    utils.scope_decrement()

def if_any(*args: List[ShaderVariable]):
    utils.append_contents(f"if({' || '.join([str(proc_bool(elem)) for elem in args])}) {'{'}\n")
    utils.scope_increment()

def if_all(*args: List[ShaderVariable]):
    utils.append_contents(f"if({' && '.join([str(proc_bool(elem)) for elem in args])}) {'{'}\n")
    utils.scope_increment()

def else_statement():
    utils.scope_decrement()
    utils.append_contents("} else {\n")
    utils.scope_increment()

def else_if_statement(arg: ShaderVariable):
    utils.scope_decrement()
    utils.append_contents(f"}} else if({proc_bool(arg)}) {'{'}\n")
    utils.scope_increment()

def else_if_any(*args: List[ShaderVariable]):
    utils.scope_decrement()
    utils.append_contents(f"}} else if({' || '.join([str(proc_bool(elem)) for elem in args])}) {'{'}\n")
    utils.scope_increment()

def else_if_all(*args: List[ShaderVariable]):
    utils.scope_decrement()
    utils.append_contents(f"}} else if({' && '.join([str(proc_bool(elem)) for elem in args])}) {'{'}\n")
    utils.scope_increment()

def return_statement(arg=None):
    arg = arg if arg is not None else ""
    utils.append_contents(f"return {arg};\n")

def while_statement(arg: ShaderVariable):
    utils.append_contents(f"while({proc_bool(arg)}) {'{'}\n")
    utils.scope_increment()

def new_scope(indent: bool = True, comment: str = None):
    if comment is None:
        utils.append_contents("{\n")
    else:
        utils.append_contents("{ " + f"/* {comment} */\n")
    
    if indent:
        utils.scope_increment()

def end(indent: bool = True):
    if indent:
        utils.scope_decrement()
        
    utils.append_contents("}\n")

def logical_and(arg1: ShaderVariable, arg2: ShaderVariable):
    return utils.new_var(dtypes.int32, f"({arg1} && {arg2})", [arg1, arg2])

def logical_or(arg1: ShaderVariable, arg2: ShaderVariable):
    return utils.new_var(dtypes.int32, f"({arg1} || {arg2})", [arg1, arg2])