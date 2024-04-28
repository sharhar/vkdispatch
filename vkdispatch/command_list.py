import vkdispatch as vd
import vkdispatch_native

import numpy as np

from typing import Callable, Any

class command_list:
    def __init__(self, reset_on_submit: bool = False) -> None:
        self._handle: int = vkdispatch_native.command_list_create(vd.get_context_handle())
        self.pc_buffers = []
        self.descriptor_sets = []
        self._reset_on_submit = reset_on_submit
    
    def __del__(self) -> None:
        pass #vkdispatch_native.command_list_destroy(self._handle)

    def get_instance_size(self) -> int:
        return vkdispatch_native.command_list_get_instance_size(self._handle)
    
    def add_pc_buffer(self, pc_buffer: 'vd.push_constant_buffer') -> None:
        self.pc_buffers.append(pc_buffer)
    
    def add_desctiptor_set(self, descriptor_set: 'vd.descriptor_set') -> None:
        self.descriptor_sets.append(descriptor_set)
    
    def reset(self) -> None:
        self.pc_buffers = []
        self.descriptor_sets = []
        vkdispatch_native.command_list_reset(self._handle)
    
    def submit(self, device_index: int = 0) -> None:
        data = b""

        for pc_buffer in self.pc_buffers:
            data += pc_buffer.get_bytes()

        if len(data) != self.get_instance_size():
            raise ValueError("Push constant buffer size mismatch!")

        vkdispatch_native.command_list_submit(self._handle, data, 1, device_index)

        if self._reset_on_submit:
            self.reset()

__cmd_list = None

def get_command_list() -> command_list:
    global __cmd_list

    if __cmd_list is None:
        __cmd_list = command_list(reset_on_submit=True)

    return __cmd_list

def get_command_list_handle() -> int:
    return get_command_list()._handle