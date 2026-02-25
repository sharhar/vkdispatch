from .backend import native

from .errors import check_for_errors

from .context import Handle
from .compute_plan import ComputePlan
from .buffer import Buffer
from .image import Sampler

from .init import log_info
from .init import is_cuda

class DescriptorSet(Handle):
    """TODO: Docstring"""
    def __init__(self, compute_plan: ComputePlan) -> None:
        super().__init__()

        self._bound_resources = []
        handle = native.descriptor_set_create(compute_plan._handle)
        check_for_errors()
        self.register_handle(handle)
        self.register_parent(compute_plan)
    
    def _destroy(self) -> None:
        native.descriptor_set_destroy(self._handle)
        check_for_errors()

    def __del__(self) -> None:
        self.destroy()

    def bind_buffer(self, buffer: Buffer, binding: int, offset: int = 0, range: int = 0, uniform: bool = False, read_access: bool = True, write_access: bool = True) -> None:
        if write_access and not getattr(buffer, "is_writable", True):
            raise ValueError("Cannot bind a read-only buffer with write access enabled.")

        self.register_parent(buffer)

        native.descriptor_set_write_buffer(
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
        self.register_parent(sampler)

        native.descriptor_set_write_image(
            self._handle,
            binding,
            sampler.image._handle,
            sampler._handle,
            1 if read_access else 0,
            1 if write_access else 0
        )
        check_for_errors()

    def set_inline_uniform_payload(self, payload: bytes) -> None:
        if not is_cuda():
            raise RuntimeError("Inline uniform payloads are currently only supported on CUDA backends.")

        native.descriptor_set_write_inline_uniform(self._handle, payload)
        check_for_errors()
