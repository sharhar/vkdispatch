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
        self._handle = vkdispatch_native.context_create(devices, queue_families)
        vd.check_for_errors()

    def __del__(self) -> None:
        pass  # vkdispatch_native.context_destroy(self._handle)

def get_compute_queue_family_index(device_index: int) -> int:
    device = vd.get_devices()[device_index]

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

def make_context(
    devices: Union[int, List[int]],
    queue_families: List[List[int]] = None
) -> Context:
    if isinstance(devices, int):
        devices = [devices]

    if queue_families is None:
        queue_families = [[get_compute_queue_family_index(dev_index)] * 2 for dev_index in devices]

    vd.initialize()

    total_devices = len(vd.get_devices())

    # Do type checking before passing to native code
    assert len(devices) == len(
        queue_families
    ), "Device and submission thread count lists must be the same length!"
    # assert all([isinstance(dev, int) for dev in devices])
    assert all(
        [type(dev) == int for dev in devices]
    ), "Device list must be a list of integers!"
    assert all(
        [dev >= 0 and dev < total_devices for dev in devices]
    ), f"All device indicies must between 0 and {total_devices}"

    return Context(devices, queue_families)


__context: Context = None


def get_context() -> Context:
    global __context

    if __context is None:
        device_count = len(vd.get_devices())
        __context = make_context([i for i in range(device_count)])

    return __context


def get_context_handle() -> int:
    return get_context()._handle
