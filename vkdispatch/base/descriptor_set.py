import vkdispatch_native

from .errors import check_for_errors

from .buffer import Buffer
from .image import Sampler

class DescriptorSet:
    """TODO: Docstring"""

    _handle: int

    def __init__(self, compute_plan_handle: int) -> None:
        self._handle = vkdispatch_native.descriptor_set_create(compute_plan_handle)
        check_for_errors()

    def bind_buffer(self, buffer: Buffer, binding: int, offset: int = 0, range: int = 0, type: int = 0) -> None:
        vkdispatch_native.descriptor_set_write_buffer(
            self._handle, binding, buffer._handle, offset, range, type
        )
        check_for_errors()

    def bind_sampler(self, sampler: Sampler, binding: int) -> None:
        vkdispatch_native.descriptor_set_write_image(self._handle, binding, sampler.image._handle, sampler._handle)
        check_for_errors()