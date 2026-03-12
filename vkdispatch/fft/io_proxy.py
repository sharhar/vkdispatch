import vkdispatch as vd
import vkdispatch.codegen as vc

from typing import List, Union, Optional

class IOProxy:
    buffer_variables: List[vc.Buffer]
    buffer_types: List[type]
    map_func: Optional[vd.MappingFunction]
    enabled: bool
    name: str

    def __init__(self, obj: Union[type, vd.MappingFunction], name: str):
        self.buffer_variables = None
        self.name = name

        if obj is None:
            self.buffer_types = []
            self.map_func = None
            self.enabled = False

        elif isinstance(obj, type):
            self.buffer_types = [vc.Buffer[obj]]
            self.map_func = None
            self.enabled = True

        elif isinstance(obj, vd.MappingFunction):
            self.buffer_types = obj.buffer_types
            self.map_func = obj
            self.enabled = True

        else:
            raise ValueError("IOObject must be initialized with a Buffer or MappingFunction")
    
    def set_variables(self, vars: List[vc.Buffer]) -> None:
        assert len(vars) == len(self.buffer_types), "Number of buffer variables does not match number of buffer types"
        if len(vars) == 0:
            self.enabled = False
            return
        
        if self.map_func is None:
            assert len(vars) == 1, "Buffer IOObject must have exactly one buffer variable"

        self.buffer_variables = vars

    def has_callback(self) -> bool:
        return self.map_func is not None

    def do_callback(self):
        assert self.map_func is not None, "IOProxy does not have a mapping function"
        self.map_func.callback(*self.buffer_variables)
