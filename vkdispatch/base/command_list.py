from typing import Any
from typing import Callable
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple
from typing import Optional

import vkdispatch_native

from .context import get_context_handle
from .errors import check_for_errors

from .compute_plan import ComputePlan
from .descriptor_set import DescriptorSet

import numpy as np

class CommandList:
    """
    A class for recording and submitting command lists to the device.

    Attributes:
        _handle (int): The handle to the command list.
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

    def record_compute_plan(self, 
                            plan: ComputePlan,
                            descriptor_set: DescriptorSet,
                            blocks: Tuple[int, int, int]) -> None:
        """
        Record a compute plan to the command list.

        Args:
            plan (ComputePlan): The compute plan to record to the command list.
            descriptor_set (DescriptorSet): The descriptor set to bind to the compute plan.
            blocks (Tuple[int, int, int]): The number of blocks to run the compute shader in.
        """

        vkdispatch_native.stage_compute_record(
            self._handle,
            plan._handle,
            descriptor_set._handle,
            blocks[0],
            blocks[1],
            blocks[2],
        )
        check_for_errors()

    def reset(self) -> None:
        """Reset the command list.
        """
        vkdispatch_native.command_list_reset(self._handle)
        check_for_errors()

    def submit(self, data: Optional[bytes] = None, stream_index: int = -2, instance_count: Optional[int] = None) -> None:
        """
        Submit the command list to the specified device with additional data to
        """

        if data is None and instance_count is None:
            raise ValueError("Data or instance count must be provided!")

        if instance_count is None:
            if len(data) == 0:
                instance_count = 1
            else:
                if len(data) % self.get_instance_size() != 0:
                    raise ValueError("Data bytes length must be a multiple of the instance size!")

                instance_count = len(data) // self.get_instance_size()
        
        if self.get_instance_size() != 0:
            assert self.get_instance_size() * instance_count == len(data), "Data length must be the product of the instance size and instance count!"

        vkdispatch_native.command_list_submit(
            self._handle, data, instance_count, [stream_index]
        )
        check_for_errors()