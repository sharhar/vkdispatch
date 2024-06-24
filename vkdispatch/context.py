from typing import List
from typing import Union

import vkdispatch as vd
import vkdispatch_native


class Context:
    """TODO: Docstring"""

    _handle: int

    def __init__(
        self,
        devices: List[int],
        queue_families: List[List[int]]
    ) -> None:
        self.devices = devices
        self.device_infos = [vd.get_devices()[dev] for dev in devices]
        self.queue_families = queue_families
        self.stream_count = sum([len(i) for i in queue_families])
        self._handle = vkdispatch_native.context_create(devices, queue_families)
        vd.check_for_errors()

    def __del__(self) -> None:
        pass  # vkdispatch_native.context_destroy(self._handle)


def get_compute_queue_family_index(device: vd.DeviceInfo, device_index: int) -> int:
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

def get_graphics_queue_family_index(device: vd.DeviceInfo, device_index: int) -> int:
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

def get_device_queues(device_index: int, max_queue_count: int) -> List[int]:
    device = vd.get_devices()[device_index]

    compute_queue_family = get_compute_queue_family_index(device, device_index)
    graphics_queue_family = get_graphics_queue_family_index(device, device_index)

    if compute_queue_family == graphics_queue_family:
        return [compute_queue_family]
    
    if "NVIDIA" in device.device_name:
        return [compute_queue_family, compute_queue_family, graphics_queue_family]

    return [compute_queue_family, graphics_queue_family]


def select_devices(use_cpu: bool) -> List[int]:
    device_infos = vd.get_devices()

    result = []

    for i, device_info in enumerate(device_infos):
        if device_info.device_type != 4 or use_cpu:
            result.append(i)

    return result


__context: Context = None

def make_context(
    devices: Union[int, List[int]] = None,
    queue_families: List[List[int]] = None,
    use_cpu: bool = False,
    max_queue_count: int = 1
) -> Context:
    global __context

    device_list = [devices]
    
    if __context is None:
        vd.initialize()
        
        if device_list[0] is None:
            device_list[0] = select_devices(use_cpu)
            
            if not queue_families is None:
                raise ValueError("If queue_families is provided, devices must also be provided!")


        if isinstance(device_list[0], int):
            device_list[0] = [device_list[0]]

        if queue_families is None:
            queue_families = [get_device_queues(dev_index, max_queue_count) for dev_index in device_list[0]]

        total_devices = len(vd.get_devices())

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

    return __context



def get_context() -> Context:
    return make_context()


def get_context_handle() -> int:
    return get_context()._handle
