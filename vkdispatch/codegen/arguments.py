import typing

from vkdispatch.codegen.variables import ShaderVariable, BufferVariable, ImageVariable
from vkdispatch.base.dtype import dtype

_ArgType = typing.TypeVar('_ArgType', bound=dtype)
_ArgCount = typing.TypeVar('_ArgCount', bound=int)

class Constant(ShaderVariable, typing.Generic[_ArgType]):
    def __init__(self) -> None:
        pass

class Variable(ShaderVariable, typing.Generic[_ArgType]):
    def __init__(self) -> None:
        pass

class ConstantArray(ShaderVariable, typing.Generic[_ArgType, _ArgCount]):
    def __init__(self) -> None:
        pass

class VariableArray(ShaderVariable, typing.Generic[_ArgType, _ArgCount]):
    def __init__(self) -> None:
        pass

class Buffer(BufferVariable, typing.Generic[_ArgType]):
    def __init__(self) -> None:
        pass

class Image1D(ImageVariable, typing.Generic[_ArgType]):
    def __init__(self) -> None:
        pass

class Image2D(ImageVariable, typing.Generic[_ArgType]):
    def __init__(self) -> None:
        pass

class Image2DArray(ImageVariable, typing.Generic[_ArgType]):
    def __init__(self) -> None:
        pass

class Image3D(ImageVariable, typing.Generic[_ArgType]):
    def __init__(self) -> None:
        pass