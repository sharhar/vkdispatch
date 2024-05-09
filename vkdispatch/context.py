from typing import List
from typing import Union

import vkdispatch
import vkdispatch_native


class Context:
    """TODO: Docstring"""

    _handle: int

    def __init__(
        self,
        devices: List[int],
        submission_thread_counts: List[int] = None,
    ) -> None:
        self._handle = vkdispatch_native.context_create(devices, submission_thread_counts)

    def __del__(self) -> None:
        pass  # vkdispatch_native.context_destroy(self._handle)


def make_context(
    devices: Union[int, List[int]],
    submission_thread_counts: Union[int, List[int]] = None,
    debug_mode: bool = False,
) -> Context:
    if isinstance(devices, int):
        devices = [devices]

    # Extend out thread counts to match the number of devices
    if submission_thread_counts is None:
        submission_thread_counts = [1] * len(devices)
    elif isinstance(submission_thread_counts, int):
        submission_thread_counts = [submission_thread_counts] * len(devices)

    vkdispatch.init_instance(debug_mode)

    total_devices = len(vkdispatch.get_devices())

    # Do type checking before passing to native code
    assert len(devices) == len(
        submission_thread_counts
    ), "Device and submission thread count lists must be the same length!"
    # assert all([isinstance(dev, int) for dev in devices])
    assert all(
        [type(dev) == int for dev in devices]
    ), "Device list must be a list of integers!"
    assert all(
        [dev >= 0 and dev < total_devices for dev in devices]
    ), f"All device indicies must between 0 and {total_devices}"

    return Context(devices, submission_thread_counts)


__context: Context = None


def get_context() -> Context:
    global __context

    if __context is None:
        device_count = len(vkdispatch.get_devices())
        __context = make_context([i for i in range(device_count)])

    return __context


def get_context_handle() -> int:
    return get_context()._handle
