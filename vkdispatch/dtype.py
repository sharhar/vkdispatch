import vkdispatch as vd

from typing import TypeVar, Generic

_NBits = TypeVar('_NBits', int)

class TypRepersentation:
    def __init__(self, var_name: str, alignment_size: int, glsl_type: str, buffer_type: str = None) -> None:
        self.var_name = var_name
        self.alginment_size = alignment_size
        self.glsl_type = glsl_type
        self.buffer_type = buffer_type if buffer_type is not None else glsl_type

    def __repr__(self) -> str:
        return f"TypRepersentation(alignment={self.alginment_size}, glsl={self.glsl_type})"

class DataType(Generic[_NBits]):
    #var_name: str
    #alginment_size: int
    #glsl_type: str
    #buffer_type: str

    def __init__(self): #, var_name: str, alignment_size: int, glsl_type: str, buffer_type: str = None) -> None:
        pass
        #self.var_name = var_name
        #self.alginment_size = alignment_size
        #self.glsl_type = glsl_type
        #self.buffer_type = buffer_type if buffer_type is not None else glsl_type
    
    @classmethod
    def __class_getitem__(cls, arg: str) -> TypRepersentation:
        return TypRepersentation(arg, 4, arg)

    #def __repr__(self) -> str:
    #    return f"DataType(alignment={self.alginment_size}, glsl={self.glsl_type})"
    
class ScalarType(DataType[_NBits]):
    def __init__(self) -> None:
        pass #super().__init__(var_name, alignment_size, glsl_type, None)
    
class VectorType(DataType):
    def __init__(self) -> None:
        pass #super().__init__(var_name, alignment_size, glsl_type, buffer_type)

class MatrixType(DataType):
    def __init__(self) -> None:
        pass #super().__init__(var_name, alignment_size, glsl_type, None)

import numpy as np

class _32Bit:
    pass

float32 = ScalarType[_32Bit]


