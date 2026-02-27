from __future__ import annotations

from . import _state as state
from ._helpers import _new_handle, _set_error, _to_bytes
from ._state import _DescriptorSet


def descriptor_set_create(plan):
    if int(plan) not in state._compute_plans:
        _set_error("Invalid compute plan handle for descriptor_set_create")
        return 0

    return _new_handle(state._descriptor_sets, _DescriptorSet(plan_handle=int(plan)))


def descriptor_set_destroy(descriptor_set):
    state._descriptor_sets.pop(int(descriptor_set), None)


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
    ds = state._descriptor_sets.get(int(descriptor_set))
    if ds is None:
        _set_error("Invalid descriptor set handle for descriptor_set_write_buffer")
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
    _set_error("CUDA Python backend does not support image objects yet")


def descriptor_set_write_inline_uniform(descriptor_set, payload):
    ds = state._descriptor_sets.get(int(descriptor_set))
    if ds is None:
        _set_error("Invalid descriptor set handle for descriptor_set_write_inline_uniform")
        return

    try:
        ds.inline_uniform_payload = _to_bytes(payload)
    except Exception as exc:
        _set_error(f"Failed to store inline uniform payload: {exc}")
