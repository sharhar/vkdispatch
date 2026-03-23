import dataclasses
import enum

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import vkdispatch as vd
import vkdispatch.codegen as vc

from ..compat import numpy_compat as npc
from vkdispatch.base.dtype import to_numpy_dtype


@dataclasses.dataclass
class BufferedStructEntry:
    memory_slice: slice
    dtype: Optional[Any]
    shape: Tuple[int, ...]


class BufferUsage(enum.Enum):
    PUSH_CONSTANT = 0
    UNIFORM_BUFFER = 1


class BufferBuilder:
    """
    A class for building buffers in memory that can be submitted to a compute pipeline.
    """

    struct_alignment: int = -1
    instance_bytes: int = 0
    instance_count: int = 0
    backing_buffer: Any = None

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
            elem_dtype = elem.dtype if elem.dtype.scalar is None else elem.dtype.scalar
            host_dtype = to_numpy_dtype(elem_dtype)

            host_shape = elem.dtype.numpy_shape

            if elem.count > 1:
                if host_shape == (1,):
                    host_shape = (elem.count,)
                else:
                    host_shape = (elem.count, *host_shape)

            element_size = npc.dtype_itemsize(host_dtype) * npc.prod(host_shape)

            self.element_map[(name, elem.name)] = BufferedStructEntry(
                slice(self.instance_bytes, self.instance_bytes + element_size),
                host_dtype,
                host_shape,
            )

            self.instance_bytes += element_size

        if self.struct_alignment != 0:
            padded_size = ((self.instance_bytes + self.struct_alignment - 1) // self.struct_alignment) * self.struct_alignment

            if padded_size != self.instance_bytes:
                self.instance_bytes = padded_size

        return offset, self.instance_bytes - offset

    def _setitem_numpy(self, key: Tuple[str, str], value: Any) -> None:
        np = npc.numpy_module()

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

    def _write_payload(self, instance_index: int, element_slice: slice, payload: bytes) -> None:
        expected_size = element_slice.stop - element_slice.start

        if len(payload) != expected_size:
            raise ValueError(f"Packed value size mismatch! Expected {expected_size}, got {len(payload)}")

        if npc.HAS_NUMPY:
            np = npc.numpy_module()
            row = self.backing_buffer[instance_index]
            row[element_slice] = np.frombuffer(payload, dtype=np.uint8)
            return

        start = instance_index * self.instance_bytes + element_slice.start
        end = start + expected_size

        self.backing_buffer[start:end] = payload

    def _pack_single_instance_value(self, value: Any, key: Tuple[str, str], buffer_element: BufferedStructEntry) -> bytes:
        expected_element_count = npc.prod(buffer_element.shape)
        flat_values = npc.flatten(value)

        if expected_element_count == 1 and len(flat_values) == 0:
            raise ValueError(f"The shape of {key} is {buffer_element.shape} but no value was given!")

        if len(flat_values) != expected_element_count:
            raise ValueError(
                f"The shape of {key} is {buffer_element.shape} but {len(flat_values)} elements were given!"
            )

        return npc.pack_values(flat_values, buffer_element.dtype)

    def _setitem_python(self, key: Tuple[str, str], value: Any) -> None:
        buffer_element = self.element_map[key]

        if self.instance_count == 1:
            payload = self._pack_single_instance_value(value, key, buffer_element)
            self._write_payload(0, buffer_element.memory_slice, payload)
            return

        # Broadcast scalar values across all instances for scalar fields.
        if not isinstance(value, (list, tuple)) and not npc.is_array_like(value) and buffer_element.shape == (1,):
            payload = self._pack_single_instance_value([value], key, buffer_element)
            for instance_index in range(self.instance_count):
                self._write_payload(instance_index, buffer_element.memory_slice, payload)
            return

        expected_element_count = npc.prod(buffer_element.shape)

        if npc.is_array_like(value):
            flat_values = npc.flatten(value)
            expected_total = expected_element_count * self.instance_count

            if len(flat_values) != expected_total:
                raise ValueError(
                    f"The shape of {key} is {(self.instance_count, *buffer_element.shape)} but {len(flat_values)} elements were given!"
                )

            for instance_index in range(self.instance_count):
                instance_values = flat_values[
                    instance_index * expected_element_count: (instance_index + 1) * expected_element_count
                ]
                payload = npc.pack_values(instance_values, buffer_element.dtype)
                self._write_payload(instance_index, buffer_element.memory_slice, payload)
            return

        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"The shape of {key} is {(self.instance_count, *buffer_element.shape)} but a scalar was given!"
            )

        if len(value) != self.instance_count:
            raise ValueError(f"Invalid shape for {key}! Expected {self.instance_count} but got {len(value)}!")

        for instance_index in range(self.instance_count):
            payload = self._pack_single_instance_value(value[instance_index], key, buffer_element)
            self._write_payload(instance_index, buffer_element.memory_slice, payload)

    def set_item(self,
                 key: Tuple[str, str],
                 value: Union[Any, list, tuple, int, float]):
        if key not in self.element_map:
            raise ValueError(f"Invalid buffer element name '{key}'!")

        if self.backing_buffer is None:
            raise RuntimeError("BufferBuilder.prepare(...) must be called before assigning values")

        buffer_element = self.element_map[key]

        if npc.HAS_NUMPY and not npc.is_host_dtype(buffer_element.dtype):
            self._setitem_numpy(key, value)
            return

        self._setitem_python(key, value)

    def __repr__(self) -> str:
        result = "Push Constant Buffer:\n"

        for key, elem in self.element_map.items():
            buffer_element = self.element_map[key]

            if npc.HAS_NUMPY and not npc.is_host_dtype(buffer_element.dtype):
                value = (self.backing_buffer[:, buffer_element.memory_slice]).view(buffer_element.dtype)
            else:
                decoded_instances = []

                for instance_index in range(self.instance_count):
                    start = instance_index * self.instance_bytes + buffer_element.memory_slice.start
                    end = instance_index * self.instance_bytes + buffer_element.memory_slice.stop
                    raw = bytes(self.backing_buffer[start:end])
                    decoded = npc.unpack_values(raw, buffer_element.dtype)
                    decoded_instances.append(decoded if len(decoded) > 1 else decoded[0])

                value = decoded_instances

            result += f"\t{key[0]}, {key[1]} ({elem.dtype}): {value}\n"

        return result[:-1]

    def prepare(self, instance_count: int) -> None:
        if self.instance_count != instance_count:
            self.instance_count = instance_count

            if npc.HAS_NUMPY:
                np = npc.numpy_module()
                self.backing_buffer = np.zeros((self.instance_count, self.instance_bytes), dtype=np.uint8)
            else:
                self.backing_buffer = bytearray(self.instance_count * self.instance_bytes)

    def toints(self):
        if npc.HAS_NUMPY:
            np = npc.numpy_module()
            return self.backing_buffer.view(np.uint32)

        return npc.from_buffer(bytes(self.backing_buffer), dtype=npc.host_dtype("uint32"), shape=(len(self.backing_buffer) // 4,))

    def tobytes(self):
        if npc.HAS_NUMPY:
            return self.backing_buffer.tobytes()

        return bytes(self.backing_buffer)
