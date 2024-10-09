import copy
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
import enum

import numpy as np

import dataclasses

import vkdispatch as vd
import vkdispatch.codegen as vc

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
    dtype: vd.dtype
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
    
    def register_element(self, name: str, dtype: vd.dtype, count: int):
        self.elements.append(StructElement(name, dtype, count))
        self.size += dtype.item_size * count

    def build(self) -> Tuple[str, List[StructElement]]:
        self.elements.sort(key=lambda x: x.dtype.item_size * x.count, reverse=True)
        return self.elements

@dataclasses.dataclass
class BufferedStructEntry:
    memory_slice: slice
    dtype: Optional[np.dtype]
    shape: Tuple[int, ...]

class BufferUsage(enum.Enum):
    PUSH_CONSTANT = 0
    UNIFORM_BUFFER = 1

class BufferBuilder:
    """TODO: Docstring"""

    struct_alignment: int = -1
    instance_bytes: int = 0
    instance_count: int = 0
    backing_buffer: np.ndarray = None

    element_map: Dict[Tuple[str, str], BufferedStructEntry] = {}

    def __init__(self, struct_alignment: Optional[int] = None, usage: Optional[BufferUsage] = None) -> None:
        assert struct_alignment is not None or usage is not None, "Either struct_alignment or usage must be provided!"

        if struct_alignment is None:
            if usage == BufferUsage.PUSH_CONSTANT:
                struct_alignment = 0
            elif usage == BufferUsage.UNIFORM_BUFFER:
                struct_alignment = vd.get_context().uniform_buffer_alignment
            else:
                raise ValueError("Invalid buffer usage!")
        
        self.struct_alignment = struct_alignment
        
    def register_struct(self, name: str, elements: List[StructElement]) -> None:
        for elem in elements:
            np_dtype = vd.to_numpy_dtype(elem.dtype if elem.dtype.scalar is None else elem.dtype.scalar)

            np_shape = elem.dtype.numpy_shape

            if elem.count > 1:
                if np_shape == (1, ):
                    np_shape = (elem.count,)
                else:
                    np_shape = (elem.count, *np_shape)

            element_size = np_dtype.itemsize * np.prod(np_shape)

            self.element_map[(name, elem.name)] = BufferedStructEntry(
                slice(self.instance_bytes, self.instance_bytes + element_size),
                np_dtype,
                np_shape
            )

            self.instance_bytes += element_size
        
        if self.struct_alignment != 0:
            padded_size = int(np.ceil(self.instance_bytes / self.struct_alignment)) * self.struct_alignment

            if padded_size != self.instance_bytes:
                self.instance_bytes = padded_size

    def __setitem__(
        self, key: Tuple[str, str], value: Union[np.ndarray, list, tuple, int, float]
    ) -> None:
        if key not in self.element_map:
            raise ValueError(f"Invalid buffer element name '{key}'!")

        buffer_element = self.element_map[key]

        if (
            not isinstance(value, np.ndarray)
            and not isinstance(value, list)
            and not isinstance(value, tuple)
            and buffer_element.shape == (1,)
        ):
            (self.backing_buffer[:, buffer_element.memory_slice]).view(buffer_element.dtype)[:] = value
            return

        arr = np.array(value, dtype=buffer_element.dtype)

        if self.instance_count != 1:
            assert arr.shape[0] == self.instance_count, f"Invalid shape for {key}! Expected {self.instance_count} but got {arr.shape[0]}!"

            if arr.shape[1:] != buffer_element.shape:
                if arr.shape != ():
                    raise ValueError(
                        f"The shape of {key} is {buffer_element.shape} but {arr.shape} was given!"
                    )
                else:
                    raise ValueError(
                        f"The shape of {key} is {buffer_element.shape} but a scalar was given!"
                    )
        
            (self.backing_buffer[:, buffer_element.memory_slice]).view(buffer_element.dtype)[:] = arr
        else:
            if arr.shape != buffer_element.shape:
                if arr.shape != ():
                    raise ValueError(
                        f"The shape of {key} is {buffer_element.shape} but {arr.shape} was given!"
                    )
                else:
                    raise ValueError(
                        f"The shape of {key} is {buffer_element.shape} but a scalar was given!"
                    )
            
            (self.backing_buffer[0, buffer_element.memory_slice]).view(buffer_element.dtype)[:] = arr

    def __repr__(self) -> str:
        result = "Push Constant Buffer:\n"

        for elem in self.elements:
            result += f"\t{elem.name} ({elem.dtype.name}): {self.numpy_arrays[elem.index]}\n"

        return result[:-1]

    def prepare(self, instance_count: int) -> None:
        if self.instance_count != instance_count:
            self.instance_count = instance_count
            self.backing_buffer = np.zeros((self.instance_count, self.instance_bytes), dtype=np.uint8)
    
    def tobytes(self):
        return self.backing_buffer.tobytes()
