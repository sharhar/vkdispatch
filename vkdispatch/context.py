import vkdispatch
import vkdispatch_native

import typing

class context:
    def __init__(self, devices: typing.List[int], submission_thread_counts: typing.List[int] = None) -> None:
        self._handle: int = vkdispatch_native.context_create(devices, submission_thread_counts)

    def __del__(self) -> None:
        pass #vkdispatch_native.context_destroy(self._handle)

def make_context(devices: typing.Union[int, typing.List[int]], submission_thread_counts: typing.Union[int, typing.List[int]] = None, debug_mode: bool = False) -> context:
    if type(devices) == int:
        devices = [devices]
    
    if submission_thread_counts is None:
        submission_thread_counts = [1] * len(devices)
    elif type(submission_thread_counts) == int:
        submission_thread_counts = [submission_thread_counts] * len(devices)

    vkdispatch.init_instance(debug_mode)
    
    total_devices = len(vkdispatch.get_devices())
    
    assert len(devices) == len(submission_thread_counts), "Device and submission thread count lists must be the same length!"
    # assert all([isinstance(dev, int) for dev in devices])
    assert all([type(dev) == int for dev in devices]), "Device list must be a list of integers!"
    assert all([dev >= 0 and dev < total_devices for dev in devices]), f"All device indicies must between 0 and {total_devices}"

    return context(devices, submission_thread_counts)

__context: context = None

def get_context() -> context:
    global __context

    if __context is None:
        device_count = len(vkdispatch.get_devices())
        __context = make_context([i for i in range(device_count)])

    return __context

def get_context_handle() -> int:
    return get_context()._handle

