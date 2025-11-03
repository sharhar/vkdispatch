from typing import Callable

__append_contents: Callable[[str], None] = None
__new_name: Callable[[], str] = None

def set_global_codegen_callbacks(append_contents: Callable[[str], None], new_name: Callable[[], str]):
    global __append_contents, __new_name
    __append_contents = append_contents
    __new_name = new_name

def append_contents(contents: str):
    global __append_contents
    __append_contents(contents)

def new_name() -> str:
    global __new_name
    return __new_name()