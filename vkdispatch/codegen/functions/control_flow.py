from ..variables.base_variable import BaseVariable

from typing import List, Optional

from . import utils

def if_statement(arg: BaseVariable, command: Optional[str] = None):
    if command is None:
        utils.append_contents(f"if({self.proc_bool(arg)}) {'{'}\n")
        self.scope_num += 1
        return
    
    self.append_contents(f"if({self.proc_bool(arg)})\n")
    self.scope_num += 1
    self.append_contents(f"{command}\n")
    self.scope_num -= 1

def if_any(*args: List[BaseVariable]):
    GlobalBuilder.obj.if_any(*args)

def if_all(*args: List[BaseVariable]):
    GlobalBuilder.obj.if_all(*args)

def else_statement():
    GlobalBuilder.obj.else_statement()

def else_if_statement(arg: BaseVariable):
    GlobalBuilder.obj.else_if_statement(arg)

def else_if_any(*args: List[BaseVariable]):
    GlobalBuilder.obj.else_if_any(*args)

def else_if_all(*args: List[BaseVariable]):
    GlobalBuilder.obj.else_if_all(*args)

def return_statement(arg=None):
    GlobalBuilder.obj.return_statement(arg)

def while_statement(arg: BaseVariable):
    GlobalBuilder.obj.while_statement(arg)

def new_scope(indent: bool = True, comment: str = None):
    GlobalBuilder.obj.new_scope(indent=indent, comment=comment)

def end(indent: bool = True):
    GlobalBuilder.obj.end(indent=indent)

def logical_and(arg1: BaseVariable, arg2: BaseVariable):
    return GlobalBuilder.obj.logical_and(arg1, arg2)

def logical_or(arg1: BaseVariable, arg2: BaseVariable):
    return GlobalBuilder.obj.logical_or(arg1, arg2)