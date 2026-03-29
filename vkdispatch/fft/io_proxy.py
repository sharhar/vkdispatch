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
        if len(vars) != len(self.buffer_types):
            raise ValueError(f"Number of buffer variables does not match number of buffer types. Expected {len(self.buffer_types)} but got {len(vars)}")
        
        if len(vars) == 0:
            self.enabled = False
            return
        
        if self.map_func is None and len(vars) != 1:
            raise ValueError("IOProxy initialized with a non-mapping function must have exactly one buffer variable")

        self.buffer_variables = vars

    def has_callback(self) -> bool:
        return self.map_func is not None

    def do_callback(self):
        if self.map_func is None:
            raise ValueError("IOProxy does not have a mapping function")
        
        self.map_func.callback(*self.buffer_variables)
