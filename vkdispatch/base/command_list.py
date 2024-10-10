from typing import Any
from typing import Callable
from typing import List
from typing import Dict
from typing import Union

import vkdispatch_native

from .context import get_context_handle
from .errors import check_for_errors

class CommandList:
    """
    A class for recording and submitting command lists to the device.
    """

    _handle: int

    def __init__(self) -> None:
        self._handle = vkdispatch_native.command_list_create(get_context_handle())
        check_for_errors()

    def __del__(self) -> None:
        pass  # vkdispatch_native.command_list_destroy(self._handle)

    def get_instance_size(self) -> int:
        """Get the total size of the command list in bytes."""
        result = vkdispatch_native.command_list_get_instance_size(self._handle)
        check_for_errors()
        return result

    def record_conditional(self) -> int:
        """Record a conditional block in the command list."""
        result = vkdispatch_native.record_conditional(self._handle)
        check_for_errors()

        return result
    
    def record_conditional_end(self) -> int:
        """Record the end of a conditional block in the command list."""
        vkdispatch_native.record_conditional_end(self._handle)
        check_for_errors()

    def reset(self) -> None:
        """Reset the command list.
        """
        vkdispatch_native.command_list_reset(self._handle)
        check_for_errors()

    # def submit(self, data: Union[bytes, None] = None, stream_index: int = -2, instance_count: int = None) -> None:
    #     """Submit the command list to the specified device with additional data to
    #     append to the front of the command list.
        
    #     Parameters:
    #     device_index (int): The device index to submit the command list to.\
    #             Default is 0.
    #     data (bytes): The additional data to append to the front of the command list.
    #     """
    #     if not self.static_constants_valid:
    #         static_data = b""
    #         for ii, uniform_buffer in enumerate(self.uniform_buffers):
    #             self.descriptor_sets[ii].bind_buffer(self.static_constant_buffer, 0, len(static_data), uniform_buffer.data_size, 1)
    #             static_data += uniform_buffer.get_bytes()

    #         if len(static_data) > 0:
    #             self.static_constant_buffer.write(static_data)
    #         self.static_constants_valid = True

    #     instances = None

    #     if data is None:
    #         data = b""

    #         for pc_buffer in self.pc_buffers:
    #             data += pc_buffer.get_bytes()

    #         instances = 1

    #         if len(data) != self.get_instance_size():
    #             raise ValueError("Push constant buffer size mismatch!")
            
    #         if instance_count is not None:
    #             instances = instance_count
    #     elif len(data) == 0:
    #         if self.get_instance_size() != 0:
    #             raise ValueError("Push constant buffer size mismatch!")

    #         instances = 1

    #         if instance_count is not None and instance_count != 1:
    #             raise ValueError("Instance count mismatch!")
    #     else:
    #         if len(data) % self.get_instance_size() != 0:
    #                 raise ValueError("Push constant buffer size mismatch!")

    #         instances = len(data) // self.get_instance_size()

    #         if instance_count is not None and instance_count != instances:
    #             raise ValueError("Instance count mismatch!")


    #     vkdispatch_native.command_list_submit(
    #         self._handle, data, instances, 1, [stream_index], False
    #     )
    #     vd.check_for_errors()

    #     if self._reset_on_submit:
    #         self.reset()
    
    # def submit_any(self, data: bytes = None, instance_count: int = None) -> None:
    #     self.submit(data=data, stream_index=-1, instance_count=instance_count)
    
    # def iter_batched_params(self, mapping_function, param_iter, batch_size: int = 10):
    #     data = b""
    #     bsize = 0

    #     for param in param_iter:
    #         mapping_function(param)

    #         for pc_buffer in self.pc_buffers:
    #             data += pc_buffer.get_bytes()

    #         bsize += 1

    #         if bsize == batch_size:
    #             yield data
    #             data = b""
    #             bsize = 0
            
    #     if bsize > 0:
    #         yield data