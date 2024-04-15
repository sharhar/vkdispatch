import vkdispatch
import vkdispatch_native

class context:
    def __init__(self, devices: list[int], submission_thread_counts: list[int] = None) -> None:
        if submission_thread_counts is None:
            submission_thread_counts = [1] * len(devices)

        self._handle = vkdispatch_native.context_create(devices, submission_thread_counts)

    def __del__(self) -> None:
        pass #vkdispatch_native.context_destroy(self._handle)