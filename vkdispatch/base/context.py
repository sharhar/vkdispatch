from typing import List
from typing import Tuple
from typing import Union
from typing import Callable

from .errors import check_for_errors
from .init import DeviceInfo, get_devices, initialize
from .dtype import float32


import vkdispatch_native

import numpy as np

class Context:
    """
    A class for managing the context of the vkdispatch library.

    Attributes:
        _handle (`int`): The handle to the context.
        devices (`List[int]`): A list of device indicies to use for the context.
        device_infos (`List[DeviceInfo]`): A list of device info objects.
        queue_families (`List[List[int]]`): A list of queue family indicies to use for the context.
        stream_count (`int`): The number of submission threads to use.
        subgroup_size (`int`): The subgroup size of the devices.
        max_workgroup_size (`Tuple[int]`): The maximum workgroup size of the devices.
        uniform_buffer_alignment (`int`): The uniform buffer alignment of the devices.
    """

    _handle: int
    devices: List[int]
    device_infos: List[DeviceInfo]
    queue_families: List[List[int]]
    stream_count: int
    subgroup_size: int
    max_workgroup_size: Tuple[int]
    uniform_buffer_alignment: int

    def __init__(
        self,
        devices: List[int],
        queue_families: List[List[int]]
    ) -> None:
        self.devices = devices
        self.device_infos = [get_devices()[dev] for dev in devices]
        self.queue_families = queue_families
        self.stream_count = sum([len(i) for i in queue_families])
        self._handle = vkdispatch_native.context_create(devices, queue_families)
        check_for_errors()
        
        subgroup_sizes = []
        max_workgroup_sizes_x = []
        max_workgroup_sizes_y = []
        max_workgroup_sizes_z = []
        uniform_buffer_alignments = []

        for device in self.device_infos:
            subgroup_sizes.append(device.sub_group_size)
            
            max_workgroup_sizes_x.append(device.max_workgroup_size[0])
            max_workgroup_sizes_y.append(device.max_workgroup_size[1])
            max_workgroup_sizes_z.append(device.max_workgroup_size[2])

            uniform_buffer_alignments.append(device.uniform_buffer_alignment)

        self.subgroup_size = min(subgroup_sizes)
        self.max_workgroup_size = (min(max_workgroup_sizes_x), min(max_workgroup_sizes_y), min(max_workgroup_sizes_z))
        self.uniform_buffer_alignment = max(uniform_buffer_alignments)

    def __del__(self) -> None:
        pass # vkdispatch_native.context_destroy(self._handle)


def get_compute_queue_family_index(device: DeviceInfo, device_index: int) -> int:
    # First check if we have a pure compute queue family with (sparse) transfer capabilities
    for i, queue_family in enumerate(device.queue_properties):
        if queue_family[1] == 6 or queue_family == 14:
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

def get_device_queues(device_index: int,  all_queues) -> List[int]:
    device = get_devices()[device_index]

    compute_queue_family = get_compute_queue_family_index(device, device_index)
    graphics_queue_family = get_graphics_queue_family_index(device, device_index)
    
    if all_queues and compute_queue_family != graphics_queue_family:
        if "NVIDIA" in device.device_name:
            return [compute_queue_family, compute_queue_family, graphics_queue_family]

        return [compute_queue_family, graphics_queue_family]

    return [compute_queue_family]

def select_devices(use_cpu: bool, all_devices) -> List[int]:
    device_infos = get_devices()

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
    
    if all_devices:
        return result
    
    return [result[0]]

__context = None
__postinit_funcs = []

def make_context(
    devices: Union[int, List[int], None] = None,
    queue_families: Union[List[List[int]], None] = None,
    use_cpu: bool = False,
    max_streams: bool = False,
    all_devices: bool = False,
    all_queues: bool = False
) -> Context:
    global __context
    global __postinit_funcs

    device_list = [devices]
    
    if __context is None:
        initialize()
        
        if device_list[0] is None:
            device_list[0] = select_devices(use_cpu, all_devices or max_streams)
            
            if not queue_families is None:
                raise ValueError("If queue_families is provided, devices must also be provided!")


        if isinstance(device_list[0], int):
            device_list[0] = [device_list[0]]

        if queue_families is None:
            queue_families = [get_device_queues(dev_index, max_streams or all_queues) for dev_index in device_list[0]]

        total_devices = len(get_devices())

        # Do type checking before passing to native code
        assert len(device_list[0]) == len(
            queue_families
        ), "Device and submission thread count lists must be the same length!"
        # assert all([isinstance(dev, int) for dev in devices])
        assert all(
            [type(dev) == int for dev in device_list[0]]
        ), "Device list must be a list of integers!"
        assert all(
            [dev >= 0 and dev < total_devices for dev in device_list[0]]
        ), f"All device indicies must between 0 and {total_devices}"

        __context = Context(device_list[0], queue_families)

        for func in __postinit_funcs:
            func()

    return __context

def is_context_initialized() -> bool:
    global __context
    return not __context is None

def stage_for_postinit(func: Callable):
    global __postinit_funcs
    __postinit_funcs.append(func)

def get_context() -> Context:
    return make_context()

def get_context_handle() -> int:
    return get_context()._handle
