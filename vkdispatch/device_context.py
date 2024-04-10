import vkdispatch
import vkdispatch_native

class DeviceContext:
    def __init__(self, devices: list[int], submission_thread_counts: list[int] = None) -> None:
        if submission_thread_counts is None:
            submission_thread_counts = [1] * len(devices)

        self._handle = vkdispatch_native.create_device_context(devices, submission_thread_counts)

    def __del__(self) -> None:
        vkdispatch_native.destroy_device_context(self._handle)