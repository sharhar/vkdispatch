import dataclasses

from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

import enum

import numpy as np

import vkdispatch as vd
import vkdispatch.codegen as vc

@dataclasses.dataclass
class BufferedStructEntry:
    memory_slice: slice
    dtype: Optional[np.dtype]
    shape: Tuple[int, ...]

class BufferUsage(enum.Enum):
    PUSH_CONSTANT = 0
    UNIFORM_BUFFER = 1

class BufferBuilder:
    """
    A class for building buffers in memory that can be submitted to a compute pipeline.

    Attributes:
        struct_alignment (int): The alignment of the struct in the buffer.
        instance_bytes (int): The size of the struct in bytes.
        instance_count (int): The number of instances of the struct.
        backing_buffer (np.ndarray): The backing buffer for the struct.
        element_map (Dict[Tuple[str, str], BufferedStructEntry]): A map of the elements in the
    """

    struct_alignment: int = -1
    instance_bytes: int = 0
    instance_count: int = 0
    backing_buffer: np.ndarray = None

    element_map: Dict[Tuple[str, str], BufferedStructEntry]

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

        self.reset()
    
    def reset(self) -> None:
        self.instance_bytes = 0
        self.instance_count = 0
        self.backing_buffer = None
        self.element_map = {}
        
    def register_struct(self, name: str, elements: List[vc.StructElement]) -> Tuple[int, int]:
        offset = self.instance_bytes

        for elem in elements:
            np_dtype = np.dtype(vd.to_numpy_dtype(elem.dtype if elem.dtype.scalar is None else elem.dtype.scalar))

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
        
        return offset, self.instance_bytes - offset

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

            if buffer_element.shape == (1,):
                arr = arr.reshape(*arr.shape, 1)

            if arr.shape[1:] != buffer_element.shape:
                if arr.shape != ():
                    raise ValueError(
                        f"The shape of {key} is {(self.instance_count, *buffer_element.shape)} but {arr.shape} was given!"
                    )
                else:
                    raise ValueError(
                        f"The shape of {key} is {buffer_element.shape} but a scalar was given!"
                    )
            
            if len(buffer_element.shape) > 1:
                (self.backing_buffer[:, buffer_element.memory_slice]).view(buffer_element.dtype).reshape(-1, *buffer_element.shape)[:] = arr
            else:
                (self.backing_buffer[:, buffer_element.memory_slice]).view(buffer_element.dtype)[:] = arr
        else:
            if arr.shape != buffer_element.shape and (len(arr.shape) > 1 and (arr.shape[0] != 1 or arr.shape[1:] != buffer_element.shape)):
                if arr.shape != ():
                    raise ValueError(
                        f"The shape of {key} is {buffer_element.shape} but {arr.shape} was given!"
                    )
                elif buffer_element.shape != (1,):
                    raise ValueError(
                        f"The shape of {key} is {buffer_element.shape} but a scalar was given!"
                    )
            if len(buffer_element.shape) > 1:
                (self.backing_buffer[0, buffer_element.memory_slice]).view(buffer_element.dtype).reshape(-1, *buffer_element.shape)[:] = arr
            else:
                (self.backing_buffer[0, buffer_element.memory_slice]).view(buffer_element.dtype)[:] = arr

#    def __repr__(self) -> str:
#        result = "Push Constant Buffer:\n"
#
#        for elem in self.elements:
#            result += f"\t{elem.name} ({elem.dtype.name}): {self.numpy_arrays[elem.index]}\n"
#
#        return result[:-1]

    def prepare(self, instance_count: int) -> None:
        if self.instance_count != instance_count:
            self.instance_count = instance_count
            self.backing_buffer = np.zeros((self.instance_count, self.instance_bytes), dtype=np.uint8)
        
    def toints(self):
        return self.backing_buffer.view(np.uint32)
    
    def tobytes(self):
        return self.backing_buffer.tobytes()
