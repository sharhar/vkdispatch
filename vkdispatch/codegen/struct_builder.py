from typing import List
from typing import Tuple

import dataclasses

import vkdispatch.base.dtype as dtype

@dataclasses.dataclass
class StructElement:
    """
    A dataclass that represents an element of a struct.

    Attributes:
       name (str): The name of the element.
       dtype (vd.dtype): The dtype of the element.
       count (int): The count of the element.
    """
    name: str
    dtype: dtype.dtype
    count: int

class StructBuilder:
    """
    A class for describing a struct in shader code (includes both the memory layout and code generation).

    Attributes:
        elements (`List[StructElement]`): A list of the elements of the struct. Given as `StructElement` objects, which contain the `name`, `index`, `dtype`, and `count` of the element.
        size (`int`): The size of the struct in bytes.
    """
    elements: List[StructElement]
    size: int

    def __init__(self, ) -> None:
        self.elements = []
        self.size = 0
    
    def register_element(self, name: str, dtype: dtype.dtype, count: int) -> None:
        self.elements.append(StructElement(name, dtype, count))
        self.size += dtype.item_size * count

    def build(self) -> List[StructElement]:
        # Sort the elements by size in descending order
        self.elements.sort(key=lambda x: x.dtype.item_size * x.count, reverse=True)
        return self.elements

