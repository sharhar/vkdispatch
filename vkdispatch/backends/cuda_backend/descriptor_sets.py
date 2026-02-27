from __future__ import annotations

from . import state as state
from .helpers import set_error, to_bytes, buffer_device_ptr

from .handle import CUDAHandle, HandleRegistry
from typing import Dict, Tuple, Optional

_descriptor_sets: HandleRegistry = HandleRegistry()

class CUDADescriptorSet(CUDAHandle):
    plan_handle: int
    buffer_bindings: Dict[int, Tuple[int, int, int, int, int, int]]
    image_bindings: Dict[int, Tuple[int, int, int, int]]
    inline_uniform_payload: bytes

    def __init__(self, plan_handle: int):
        super().__init__(_descriptor_sets)

        self.plan_handle = plan_handle
        self.buffer_bindings = {}
        self.image_bindings = {}
        self.inline_uniform_payload = b""

    @staticmethod
    def from_handle(handle: int) -> Optional["CUDADescriptorSet"]:
        return _descriptor_sets.get(int(handle))
    
    def resolve_buffer_pointer(self, binding: int) -> int:
        binding_info = self.buffer_bindings.get(binding)
        if binding_info is None:
            raise RuntimeError(f"Missing descriptor buffer binding {binding}")

        buffer_handle, offset, _, _, _, _ = binding_info

        buffer_obj = state.buffers.get(int(buffer_handle))
        if buffer_obj is None:
            raise RuntimeError(f"Invalid buffer handle {buffer_handle} for binding {binding}")

        return buffer_device_ptr(buffer_obj) + int(offset)

def descriptor_set_create(plan):
    if int(plan) not in state.compute_plans:
        set_error("Invalid compute plan handle for descriptor_set_create")
        return 0

    return CUDADescriptorSet(plan_handle=int(plan)).handle


def descriptor_set_destroy(descriptor_set):
    _descriptor_sets.pop(descriptor_set)


def descriptor_set_write_buffer(
    descriptor_set,
    binding,
    object,
    offset,
    range,
    uniform,
    read_access,
    write_access,
):
    ds = CUDADescriptorSet.from_handle(descriptor_set)
    if ds is None:
        set_error("Invalid descriptor set handle for descriptor_set_write_buffer")
        return

    ds.buffer_bindings[int(binding)] = (
        int(object),
        int(offset),
        int(range),
        int(uniform),
        int(read_access),
        int(write_access),
    )


def descriptor_set_write_image(
    descriptor_set,
    binding,
    object,
    sampler_obj,
    read_access,
    write_access,
):
    _ = descriptor_set
    _ = binding
    _ = object
    _ = sampler_obj
    _ = read_access
    _ = write_access
    set_error("CUDA Python backend does not support image objects yet")


def descriptor_set_write_inline_uniform(descriptor_set, payload):
    ds = CUDADescriptorSet.from_handle(descriptor_set)
    if ds is None:
        set_error("Invalid descriptor set handle for descriptor_set_write_inline_uniform")
        return

    try:
        ds.inline_uniform_payload = to_bytes(payload)
    except Exception as exc:
        set_error(f"Failed to store inline uniform payload: {exc}")
