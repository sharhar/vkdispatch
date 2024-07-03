import typing
from vkdispatch.types import bits
from vkdispatch.types import dimentions
from vkdispatch.types import numbers

class BaseType(typing.Generic[bits._NBits, numbers._Number, dimentions._NDims]):
    nbits: bits._NBits
    number: numbers._Number
    dims: dimentions._NDims

    def __init__(self) -> None:
        super().__init__()

        self.nbits = bits._NBits()
        self.number = numbers._Number()
        self.dims = dimentions._NDims()