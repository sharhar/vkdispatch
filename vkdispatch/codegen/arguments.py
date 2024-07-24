import typing


from vkdispatch.codegen.variables import ShaderVariable, BufferVariable, ImageVariable
from vkdispatch.dtype import dtype

_ArgType = typing.TypeVar('_ArgType', bound=dtype)

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