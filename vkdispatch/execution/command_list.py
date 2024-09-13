from typing import Any
from typing import Callable
from typing import List
from typing import Dict
from typing import Union

import copy

import numpy as np

import vkdispatch as vd
import vkdispatch.codegen as vc
import vkdispatch_native

class CommandList:
    """TODO: Docstring"""

    _handle: int
    _reset_on_submit: bool
    submit_on_record: bool
    pc_buffers: "List[vc.BufferStructureProxy]"
    uniform_buffers: "List[vc.BufferStructureProxy]"
    descriptor_sets: "List[vd.DescriptorSet]"

    def __init__(self, reset_on_submit: bool = False, submit_on_record: bool = False) -> None:
        self._handle = vkdispatch_native.command_list_create(vd.get_context_handle())
        vd.check_for_errors()
        self.pc_buffers = []
        self.uniform_buffers = []
        self.descriptor_sets = []
        self._reset_on_submit = reset_on_submit
        self.submit_on_record = submit_on_record

        self.static_constants_size = 0
        self.static_constants_valid = False
        self.static_constant_buffer = vd.Buffer(shape=(4096,), var_type=vd.uint32) # Create a base static constants buffer at size 4k bytes

    def __del__(self) -> None:
        pass  # vkdispatch_native.command_list_destroy(self._handle)

    def get_instance_size(self) -> int:
        """Get the total size of the command list in bytes."""
        result = vkdispatch_native.command_list_get_instance_size(self._handle)
        vd.check_for_errors()
        return result

    def add_pc_buffer(self, pc_buffer: "vd.BufferStructureProxy") -> None:
        """Add a push constant buffer to the command list."""
        pc_buffer.index = len(self.pc_buffers)
        self.pc_buffers.append(pc_buffer)

    def add_desctiptor_set_and_static_constants(self, descriptor_set: "vd.DescriptorSet", constants_buffer_proxy: "vc.BufferStructureProxy") -> None:
        """Add a descriptor set to the command list."""
        self.descriptor_sets.append(descriptor_set)

        self.static_constants_size += constants_buffer_proxy.size
        self.uniform_buffers.append(constants_buffer_proxy)

        if(self.static_constants_size > self.static_constant_buffer.size):
            new_size = int(np.ceil(self.static_constants_size / 4))
            self.static_constant_buffer = vd.Buffer(shape=(new_size,), var_type=vd.uint32)
        
        self.static_constants_valid = False

    def reset(self) -> None:
        """Reset the command list by clearing the push constant buffer and descriptor
        set lists. The call to command_list_reset frees all associated memory.
        """
        self.pc_buffers = []
        self.descriptor_sets = []
        self.uniform_buffers = []

        self.static_constants_size = 0
        self.static_constants_valid = False

        vkdispatch_native.command_list_reset(self._handle)
        vd.check_for_errors()

    def submit(self, data: Union[bytes, None] = None, stream_index: int = -2, instance_count: int = None) -> None:
        """Submit the command list to the specified device with additional data to
        append to the front of the command list.
        
        Parameters:
        device_index (int): The device index to submit the command list to.\
                Default is 0.
        data (bytes): The additional data to append to the front of the command list.
        """
        if not self.static_constants_valid:
            static_data = b""
            for ii, uniform_buffer in enumerate(self.uniform_buffers):
                self.descriptor_sets[ii].bind_buffer(self.static_constant_buffer, 0, len(static_data), uniform_buffer.data_size, 1)
                static_data += uniform_buffer.get_bytes()

            if len(static_data) > 0:
                self.static_constant_buffer.write(static_data)
            self.static_constants_valid = True

        instances = None

        if data is None:
            data = b""

            for pc_buffer in self.pc_buffers:
                data += pc_buffer.get_bytes()

            instances = 1

            if len(data) != self.get_instance_size():
                raise ValueError("Push constant buffer size mismatch!")
            
            if instance_count is not None:
                instances = instance_count
        elif len(data) == 0:
            if self.get_instance_size() != 0:
                raise ValueError("Push constant buffer size mismatch!")

            instances = 1

            if instance_count is not None and instance_count != 1:
                raise ValueError("Instance count mismatch!")
        else:
            if len(data) % self.get_instance_size() != 0:
                    raise ValueError("Push constant buffer size mismatch!")

            instances = len(data) // self.get_instance_size()

            if instance_count is not None and instance_count != instances:
                raise ValueError("Instance count mismatch!")


        vkdispatch_native.command_list_submit(
            self._handle, data, instances, 1, [stream_index], False
        )
        vd.check_for_errors()

        if self._reset_on_submit:
            self.reset()
    
    def submit_any(self, data: bytes) -> None:
        self.submit(data=data, stream_index=-1)
    
    def iter_batched_params(self, mapping_function, param_iter, batch_size: int = 10):
        data = b""
        bsize = 0

        for param in param_iter:
            mapping_function(param)

            for pc_buffer in self.pc_buffers:
                data += pc_buffer.get_bytes()

            bsize += 1

            if bsize == batch_size:
                yield data
                data = b""
                bsize = 0
            
        if bsize > 0:
            yield data

__default_cmd_list = None
__custom_list = None

def default_cmd_list() -> CommandList:
    global __default_cmd_list

    if __default_cmd_list is None:
        __default_cmd_list = CommandList(reset_on_submit=True, submit_on_record=True)

    return __default_cmd_list

def global_cmd_list() -> CommandList:
    global __custom_list

    if __custom_list is not None:
        return __custom_list

    return default_cmd_list()

def set_global_cmd_list(cmd_list: CommandList = None) -> CommandList:
    global __custom_list
    old_value = __custom_list
    __custom_list = cmd_list 
    return old_value