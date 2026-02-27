from typing import Tuple
from typing import Optional

from ..backends.backend_selection import native
from .init import is_cuda

from .context import Handle
from .errors import check_for_errors

from ..execution_pipeline.cuda_graph_capture import get_cuda_capture

from .compute_plan import ComputePlan
from .descriptor_set import DescriptorSet

class CommandList(Handle):
    """
    Represents a sequence of GPU commands to be executed on a device.

    CommandLists are used to record dispatch operations, memory barriers, and 
    synchronization points. They act as the primary unit of work submission 
    to the Vulkan queue.

    Attributes:
        _handle (int): The internal handle to the native Vulkan command buffer wrapper.
    """

    def __init__(self) -> None:
        super().__init__()

        handle = native.command_list_create(self.context._handle)
        self.register_handle(handle)
        check_for_errors()

    def _destroy(self) -> None:
        native.command_list_destroy(self._handle)
        check_for_errors()

    def __del__(self) -> None:
        self.destroy()

    def get_instance_size(self) -> int:
        """Get the total size of the command list in bytes."""
        result = native.command_list_get_instance_size(self._handle)
        check_for_errors()
        return result

    def record_compute_plan(self, 
                            plan: ComputePlan,
                            descriptor_set: DescriptorSet,
                            blocks: Tuple[int, int, int]) -> None:
        """
        Records a compute shader dispatch into the command list.

        :param plan: The compiled compute plan (shader) to execute.
        :type plan: vkdispatch.base.compute_plan.ComputePlan
        :param descriptor_set: The resource bindings (buffers, images) for this execution.
        :type descriptor_set: vkdispatch.base.descriptor_set.DescriptorSet
        :param blocks: The dimensions of the workgroup grid (x, y, z) to dispatch.
        :type blocks: Tuple[int, int, int]
        """
        self.register_parent(plan)
        self.register_parent(descriptor_set)

        native.stage_compute_record(
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
        native.command_list_reset(self._handle)
        check_for_errors()

        self.clear_parents()

    def submit(
        self,
        data: Optional[bytes] = None,
        queue_index: int = -2,
        instance_count: Optional[int] = None,
        cuda_stream=None
    ) -> None:
        """
        Submits the recorded command list to the GPU queue for execution.

        :param data: Optional binary data (e.g., push constants) to append to the 
                     front of the command list buffer before submission.
        :type data: Optional[bytes]
        :param queue_index: The index of the queue to submit to. -2 uses the default queue associated 
                            with the command list's context.
        :type queue_index: int
        :param instance_count: The number of instances to execute if instanced dispatch is used.
        :type instance_count: Optional[int]
        :raises ValueError: If data length logic conflicts with instance size.
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

        if cuda_stream is None and get_cuda_capture() is not None:
            cuda_stream = get_cuda_capture().cuda_stream

        if cuda_stream is not None:
            if not is_cuda():
                raise RuntimeError("cuda_stream=... is currently only supported with CUDA backends.")

            native.cuda_stream_override_begin(cuda_stream)
            check_for_errors()

        done = False
        while not done:
            done = native.command_list_submit(
                self._handle, data, instance_count, queue_index
            )
            check_for_errors()

        if cuda_stream is not None:
            native.cuda_stream_override_end()
