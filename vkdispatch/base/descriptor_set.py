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

    def bind_buffer(self, buffer: Buffer, binding: int, offset: int = 0, range: int = 0, uniform: bool = False, read_access: bool = True, write_access: bool = True) -> None:
        vkdispatch_native.descriptor_set_write_buffer(
            self._handle,
            binding,
            buffer._handle,
            offset,
            range,
            1 if uniform else 0,
            1 if read_access else 0,
            1 if write_access else 0
        )
        check_for_errors()

    def bind_sampler(self, sampler: Sampler, binding: int, read_access: bool = True, write_access: bool = True) -> None:
        vkdispatch_native.descriptor_set_write_image(
            self._handle,
            binding,
            sampler.image._handle,
            sampler._handle,
            1 if read_access else 0,
            1 if write_access else 0
        )
        check_for_errors()