import copy
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

import vkdispatch as vd

class BufferStructureProxy:
    """TODO: Docstring"""

    pc_dict: Dict[str, Tuple[int, vd.dtype, int]]
    ref_dict: Dict[str, int]
    pc_list: List[np.ndarray]
    var_types: List[vd.dtype]
    numpy_dtypes: List
    size: int
    data_size: int
    prologue: bytes
    index: int
    alignment: int

    def __init__(self, pc_dict: dict, alignment: int) -> None:
        self.pc_dict = copy.deepcopy(pc_dict)
        self.ref_dict = {}
        self.pc_list = [None] * len(self.pc_dict)
        self.var_types = [None] * len(self.pc_dict)
        self.numpy_dtypes = [None] * len(self.pc_dict)
        self.data_size = 0
        self.prologue = b""
        self.size = 0
        self.index = 0
        self.alignment = alignment

        #uniform_alignment = vd.get_context().device_infos[0].uniform_buffer_alignment
        #self.static_constants_size = int(np.ceil(uniform_buffer.size / float(uniform_alignment))) * int(uniform_alignment)

        # Populate the push constant buffer with the given dictionary
        for key, val in self.pc_dict.items():
            ii, var_type, count = val

            self.ref_dict[key] = ii

            dtype = vd.to_numpy_dtype(var_type if var_type.scalar is None else var_type.scalar)
            self.numpy_dtypes[ii] = dtype
            self.pc_list[ii] = np.zeros(
                shape=var_type.numpy_shape, dtype=self.numpy_dtypes[ii]
            )
            self.var_types[ii] = var_type

            self.data_size += var_type.item_size * count
        
        if self.alignment == 0:
            self.size = self.data_size
        else:
            self.size = int(np.ceil(self.data_size / self.alignment)) * self.alignment

            self.prologue = b"\x00" * (self.size - self.data_size)

    def __setitem__(
        self, key: str, value: Union[np.ndarray, list, tuple, int, float]
    ) -> None:
        if key not in self.ref_dict:
            raise ValueError(f"Invalid push constant '{key}'!")

        ii = self.ref_dict[key]

        if (
            not isinstance(value, np.ndarray)
            and not isinstance(value, list)
            and not isinstance(value, tuple)
            and self.pc_list[ii].shape == (1,)
        ):
            self.pc_list[ii][0] = value
            return

        arr = np.array(value, dtype=self.numpy_dtypes[ii])

        if arr.shape != self.var_types[ii].numpy_shape:
            if arr.shape != ():
                raise ValueError(
                    f"The shape of {key} is {self.var_types[ii].numpy_shape} but {arr.shape} was given!"
                )
            else:
                raise ValueError(
                    f"The shape of {key} is {self.var_types[ii].numpy_shape} but a scalar was given!"
                )

        self.pc_list[ii] = arr


    def __repr__(self) -> str:
        result = "Push Constant Buffer:\n"

        for key, val in self.pc_dict.items():
            ii, var_type = val
            result += f"\t{key} ({var_type.name}): {self.pc_list[ii]}\n"

        return result[:-1]

    def get_bytes(self):
        return b"".join([elem.tobytes() for elem in self.pc_list]) + self.prologue


class BufferStructure:
    my_dict: Dict[str, Tuple[int, vd.dtype]]
    my_list: List[Tuple[str, vd.dtype, str]]
    my_size: int

    def __init__(self, ) -> None:
        self.my_dict = {}
        self.my_list = []
        self.my_size = 0
    
    def register_element(self, var_name: str, var_type: vd.dtype, var_decleration: str, count: int):
        self.my_list.append((var_name, var_type, var_decleration, count))
        self.my_size += var_type.item_size * count

    def build(self) -> Tuple[str, Dict[str, Tuple[int, vd.dtype, int]]]:
        self.my_list.sort(key=lambda x: x[1].item_size * x[3], reverse=True)
        self.my_dict = {elem[0]: (ii, elem[1], elem[3]) for ii, elem in enumerate(self.my_list)}

        if len(self.my_list) == 0:
            return "", None

        buffer_decleration_contents = "\n".join(
            [f"\t{elem[2]}" for elem in self.my_list]
        )

        return buffer_decleration_contents, self.my_dict