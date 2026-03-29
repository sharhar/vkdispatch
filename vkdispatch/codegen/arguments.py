import typing

from .variables.variables import ShaderVariable
from .variables.bound_variables import BufferVariable, ImageVariable
from vkdispatch.base.dtype import dtype

_ArgType = typing.TypeVar('_ArgType', bound=dtype)
_ArgCount = typing.TypeVar('_ArgCount', bound=int)

class Constant(ShaderVariable, typing.Generic[_ArgType]):
    def __init__(self) -> None:
        pass

class Variable(ShaderVariable, typing.Generic[_ArgType]):
    def __init__(self) -> None:
        pass

class Buffer(BufferVariable, typing.Generic[_ArgType]):
    def __init__(self) -> None:
        pass

class Image1D(ImageVariable, typing.Generic[_ArgType]):
    dimensions: int = 1

    def __init__(self) -> None:
        pass

class Image2D(ImageVariable, typing.Generic[_ArgType]):
    dimensions: int = 2

    def __init__(self) -> None:
        pass

class Image2DArray(ImageVariable, typing.Generic[_ArgType]):
    dimensions: int = 2

    def __init__(self) -> None:
        pass

class Image3D(ImageVariable, typing.Generic[_ArgType]):
    dimensions: int = 3

    def __init__(self) -> None:
        pass