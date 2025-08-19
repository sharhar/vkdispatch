import vkdispatch_native

from typing import List, Union

from .errors import check_for_errors

from .context import get_context, Context
from .compute_plan import ComputePlan
from .buffer import Buffer
from .image import Sampler

class DescriptorSet:
    """TODO: Docstring"""

    context: Context
    _handle: int
    _compute_plan: ComputePlan
    _bound_resources: List[Union[Buffer, Sampler]]

    def __init__(self, compute_plan: ComputePlan) -> None:
        self._bound_resources = []
        self._compute_plan = compute_plan
        self.context = get_context()

        self._handle = vkdispatch_native.descriptor_set_create(compute_plan._handle)
        check_for_errors()

    def __del__(self) -> None:
        vkdispatch_native.descriptor_set_destroy(self._handle)

    def bind_buffer(self, buffer: Buffer, binding: int, offset: int = 0, range: int = 0, uniform: bool = False, read_access: bool = True, write_access: bool = True) -> None:
        self._bound_resources.append(buffer)
        
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
        self._bound_resources.append(sampler)

        vkdispatch_native.descriptor_set_write_image(
            self._handle,
            binding,
            sampler.image._handle,
            sampler._handle,
            1 if read_access else 0,
            1 if write_access else 0
        )
        check_for_errors()