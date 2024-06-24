import vkdispatch as vd
import vkdispatch_native


class DescriptorSet:
    """TODO: Docstring"""

    _handle: int

    def __init__(self, compute_plan_handle: int) -> None:
        self._handle = vkdispatch_native.descriptor_set_create(compute_plan_handle)
        vd.check_for_errors()

    def bind_buffer(self, buffer: vd.Buffer, binding: int, offset: int = 0, range: int = 0, type: int = 0) -> None:
        vkdispatch_native.descriptor_set_write_buffer(
            self._handle, binding, buffer._handle, offset, range, type
        )
        vd.check_for_errors()
