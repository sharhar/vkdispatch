from typing import List
from typing import Tuple
from typing import Union
from typing import Dict
from typing import MutableMapping

import atexit
import weakref

from .errors import check_for_errors
from .init import DeviceInfo, get_devices, initialize

import vkdispatch_native

class Handle:
    context: "Context"
    _handle: int
    destroyed: bool

    parents: Dict[int, "Handle"]
    children_dict: MutableMapping[int, "Handle"]

    def __init__(self):
        self.context = get_context()
        self._handle = None
        self.destroyed = False
        self.parents = {}
        self.children_dict = weakref.WeakValueDictionary()

    def register_handle(self, handle: int) -> None:
        """
        Registers the handle in the context's handles dictionary.
        """
        self._handle = handle
        self.context.handles_dict[self._handle] = self

    def register_parent(self, parent: "Handle") -> None:
        """
        Registers the parent handle.
        """
        if parent._handle in self.parents.keys():
            return
        
        self.parents[parent._handle] = parent

        parent.add_child_handle(self)
    
    def clear_parents(self) -> None:
        """
        Clears the parent handles.
        """
        for parent in self.parents.values():
            parent.remove_child_handle(self)

        self.parents.clear()

    def add_child_handle(self, child: "Handle") -> None:
        """
        Adds a child handle to the current handle.
        """
        if child._handle in self.children_dict.keys():
            raise ValueError(f"Child handle {child._handle} already exists in parent handle!")
        
        self.children_dict[child._handle] = child

    def remove_child_handle(self, child: "Handle") -> None:
        """
        Removes a child handle from the current handle.
        """
        if child._handle not in self.children_dict.keys():
            raise ValueError(f"Child handle {child._handle} does not exist in parent handle!")
        
        self.children_dict.pop(child._handle)

    def _destroy(self) -> None:
        raise NotImplementedError("destroy is an abstract method and must be implemented by subclasses.")
    
    def destroy(self) -> None:
        """
        Destroys the context handle and cleans up resources.
        """
        if self.destroyed:
            return

        child_list = list(self.children_dict.values())

        for child in child_list:
            child.destroy()

        assert len(self.children_dict) == 0, "Not all children were destroyed!"
        
        self._destroy()
        check_for_errors()
        
        self.clear_parents()

        if self._handle in self.context.handles_dict.keys():
            self.context.handles_dict.pop(self._handle)
        
        self.destroyed = True

class Context:
    """
    A class for managing the context of the vkdispatch library.

    Attributes:
        _handle (`int`): The handle to the context.
        devices (`List[int]`): A list of device indicies to use for the context.
        device_infos (`List[DeviceInfo]`): A list of device info objects.
        queue_families (`List[List[int]]`): A list of queue family indicies to use for the context.
        queue_count (`int`): The number of queues in the context.
        subgroup_size (`int`): The subgroup size of the devices.
        max_workgroup_size (`Tuple[int]`): The maximum workgroup size of the devices.
        max_workgroup_invocations (`int`): The maximum number of workgroup invocations.
        max_workgroup_count (`Tuple[int, int, int]`): The maximum workgroup count of the devices.
        uniform_buffer_alignment (`int`): The uniform buffer alignment of the devices.
        max_shared_memory (`int`): The maximum shared memory size of the devices.
    """

    _handle: int
    devices: List[int]
    device_infos: List[DeviceInfo]
    queue_families: List[List[int]]
    queue_count: int
    subgroup_size: int
    max_workgroup_size: Tuple[int]
    max_workgroup_invocations: int
    max_workgroup_count: Tuple[int, int, int]
    uniform_buffer_alignment: int
    max_shared_memory: int
    handles_dict: MutableMapping[int, Handle]

    def __init__(
        self,
        devices: List[int],
        queue_families: List[List[int]]
    ) -> None:
        self.devices = devices
        self.device_infos = [get_devices()[dev] for dev in devices]
        self.queue_families = queue_families
        self.queue_count = sum([len(i) for i in queue_families])
        self.handles_dict = weakref.WeakValueDictionary()
        self._handle = vkdispatch_native.context_create(devices, queue_families)
        check_for_errors()
        
        subgroup_sizes = []
        max_workgroup_sizes_x = []
        max_workgroup_sizes_y = []
        max_workgroup_sizes_z = []
        max_workgroup_invocations = []
        max_workgroup_counts_x = []
        max_workgroup_counts_y = []
        max_workgroup_counts_z = []
        uniform_buffer_alignments = []
        max_shared_memory = []

        for device in self.device_infos:
            subgroup_sizes.append(device.sub_group_size)
            
            max_workgroup_sizes_x.append(device.max_workgroup_size[0])
            max_workgroup_sizes_y.append(device.max_workgroup_size[1])
            max_workgroup_sizes_z.append(device.max_workgroup_size[2])

            max_workgroup_invocations.append(device.max_workgroup_invocations)

            max_workgroup_counts_x.append(device.max_workgroup_count[0])
            max_workgroup_counts_y.append(device.max_workgroup_count[1])
            max_workgroup_counts_z.append(device.max_workgroup_count[2])

            uniform_buffer_alignments.append(device.uniform_buffer_alignment)

            max_shared_memory.append(device.max_compute_shared_memory_size)

        self.subgroup_size = min(subgroup_sizes)
        self.max_workgroup_size = (
            min(max_workgroup_sizes_x),
            min(max_workgroup_sizes_y),
            min(max_workgroup_sizes_z)
        )
        
        self.max_workgroup_invocations = min(max_workgroup_invocations)
        self.max_workgroup_count = (
            min(max_workgroup_counts_x),
            min(max_workgroup_counts_y),
            min(max_workgroup_counts_z)
        )

        self.uniform_buffer_alignment = max(uniform_buffer_alignments)
        self.max_shared_memory = min(max_shared_memory)

def get_compute_queue_family_index(device: DeviceInfo, device_index: int) -> int:
    # First check if we have a pure compute queue family with (sparse) transfer capabilities
    for i, queue_family in enumerate(device.queue_properties):
        if queue_family[1] == 6 or queue_family[1] == 14:
            return i

    # If not, check if we have a compute queue family without graphics capabilities
    for i, queue_family in enumerate(device.queue_properties):
        if queue_family[1] & 2 and not queue_family[1] & 1:
            return i
        
    # Finnally, return any queue with compute capabilities
    for i, queue_family in enumerate(device.queue_properties):
        if queue_family[1] & 2:
            return i

    raise ValueError(f"Device {device_index} does not have a compute queue family!")

def get_graphics_queue_family_index(device: DeviceInfo, device_index: int) -> int:
    # First check if we have a pure compute queue family with (sparse) transfer capabilities
    for i, queue_family in enumerate(device.queue_properties):
        if queue_family[1] == 7 or queue_family == 15:
            return i

    # If not, check if we have a compute queue family without graphics capabilities
    for i, queue_family in enumerate(device.queue_properties):
        if queue_family[1] & 2 and queue_family[1] & 1:
            return i
        
    # Finnally, return any queue with compute capabilities
    for i, queue_family in enumerate(device.queue_properties):
        if queue_family[1] & 2:
            return i

    raise ValueError(f"Device {device_index} does not have a compute queue family!")

def select_devices(use_cpu: bool, device_count) -> List[int]:
    device_infos = get_devices()

    if device_count is not None:
        if device_count < 0 or device_count > len(device_infos):
            raise ValueError(f"Device count must be between 0 and {len(device_infos)}")

    result = []

    # Check for Discrete GPU (Type 2)
    for i, device_info in enumerate(device_infos):
        if device_info.device_type == 2:
            result.append(i)
    
    # Check for Integrated GPU (Type 1)
    for i, device_info in enumerate(device_infos):
        if device_info.device_type == 1:
            result.append(i)
    
    # Check for Virtual GPU (Type 3)
    for i, device_info in enumerate(device_infos):
        if device_info.device_type == 3:
            result.append(i)

    # Check for CPU (Type 4)
    if use_cpu:
        for i, device_info in enumerate(device_infos):
            if device_info.device_type == 4:
                result.append(i)
    
    return result[:device_count] if device_count is not None else result

__context = None

def select_queue_families(device_index: int, queue_count: int = None) -> List[int]:
    device = get_devices()[device_index]

    compute_queue_family = get_compute_queue_family_index(device, device_index)
    graphics_queue_family = get_graphics_queue_family_index(device, device_index)

    if queue_count is None:
        queue_count = 2

        if device.is_nvidia():
            queue_count = 3

        if compute_queue_family == graphics_queue_family:
            queue_count = 1
    
    queue_families = []

    for i in range(queue_count):
        # For NVIDIA, it's better to just have one graphics queue family
        # and mulitple compute queues
        if device.is_nvidia() and i != 1:
            queue_families.append(compute_queue_family)
            continue

        if i % 2 == 0:
            queue_families.append(compute_queue_family)
        else:
            queue_families.append(graphics_queue_family)

    return queue_families

def make_context(
    device_ids: Union[int, List[int], None] = None,
    device_count: Union[int, None] = None,
    queue_counts: Union[int, List[int], None] = None,
    queue_families: Union[List[List[int]], None] = None,
    use_cpu: bool = False,
    multi_device: bool = False,
    multi_queue: bool = False,
) -> Context:
    global __context
    
    if __context is None:
        initialize()
        
        if device_ids is None:
            if device_count is None:
                device_count = None if multi_device else 1

            device_ids = select_devices(use_cpu, device_count)
            
            if not queue_families is None:
                raise ValueError("If queue_families is provided, devices must also be provided!")

        if isinstance(device_ids, int):
            device_ids = [device_ids]
        
        if queue_families is None:
            queue_families = []

            for ii, dev_index in enumerate(device_ids):
                queue_family_count = None if multi_queue else 1

                if queue_counts is not None:
                    if isinstance(queue_counts, int):
                        queue_family_count = queue_counts
                    else:
                        queue_family_count = queue_counts[ii]

                queue_families.append(
                    select_queue_families(dev_index, queue_family_count)
                )

        total_devices = len(get_devices())

        # Do type checking before passing to native code
        assert len(device_ids) == len(
            queue_families
        ), "Device and submission thread count lists must be the same length!"
        
        assert all(
            [type(dev) == int for dev in device_ids]
        ), "Device list must be a list of integers!"

        assert all(
            [dev >= 0 and dev < total_devices for dev in device_ids]
        ), f"All device indicies must between 0 and {total_devices}"

        print(f"Creating context with devices {device_ids} and queue families {queue_families}")

        __context = Context(device_ids, queue_families)

    return __context

def is_context_initialized() -> bool:
    global __context
    return not __context is None

def get_context() -> Context:
    return make_context()

def get_context_handle() -> int:
    return get_context()._handle

def queue_wait_idle(queue_index: int = None) -> None:
    """
    Wait for the specified queue to finish processing. For all queues, leave queue_index as None.
    
    Args:
        queue_index (int): The index of the queue.
    """

    assert queue_index is None or isinstance(queue_index, int), "queue_index must be an integer or None."
    assert queue_index is None or queue_index >= -1, "queue_index must be a non-negative integer or -1 (for all queues)."
    assert queue_index is None or queue_index < get_context().queue_count, f"Queue index {queue_index} is out of bounds for context with {get_context().queue_count} queues."

    vkdispatch_native.context_queue_wait_idle(get_context_handle(), queue_index if queue_index is not None else -1)
    check_for_errors()


def destroy_context() -> None:
    """
    Destroys the current context and cleans up resources.
    """
    global __context

    if __context is not None:
        handles_list = list(__context.handles_dict.values())

        for handle in handles_list:
            handle.destroy()

        assert len(__context.handles_dict) == 0, "Not all handles were destroyed!"

        vkdispatch_native.context_destroy(__context._handle)
        __context = None

atexit.register(destroy_context)
