from __future__ import annotations

import hashlib

from . import _state as state
from ._cuda_primitives import cuda
from ._helpers import (
    _activate_context,
    _clear_error,
    _coerce_stream_handle,
    _new_handle,
    _query_max_kernel_param_size,
    _set_error,
    _stream_override_stack,
)
from ._state import _Context


def init(debug, log_level):
    state._debug_mode = bool(debug)
    state._log_level = int(log_level)
    _clear_error()

    if state._initialized:
        return

    cuda.init()
    state._initialized = True


def log(log_level, text, file_str, line_str):
    _ = log_level
    _ = text
    _ = file_str
    _ = line_str


def set_log_level(log_level):
    state._log_level = int(log_level)


def get_devices():
    if not state._initialized:
        init(False, state._log_level)

    try:
        device_count = cuda.Device.count()
    except Exception as exc:
        _set_error(f"Failed to enumerate CUDA devices: {exc}")
        return []

    driver_version = 0
    try:
        driver_version = int(cuda.get_driver_version())
    except Exception:
        driver_version = 0

    devices = []

    for index in range(device_count):
        dev = cuda.Device(index)
        attrs = dev.get_attributes()
        cc_major, cc_minor = dev.compute_capability()
        total_memory = int(dev.total_memory())

        max_workgroup_size = (
            int(attrs.get(cuda.device_attribute.MAX_BLOCK_DIM_X, 0)),
            int(attrs.get(cuda.device_attribute.MAX_BLOCK_DIM_Y, 0)),
            int(attrs.get(cuda.device_attribute.MAX_BLOCK_DIM_Z, 0)),
        )

        max_workgroup_count = (
            int(attrs.get(cuda.device_attribute.MAX_GRID_DIM_X, 0)),
            int(attrs.get(cuda.device_attribute.MAX_GRID_DIM_Y, 0)),
            int(attrs.get(cuda.device_attribute.MAX_GRID_DIM_Z, 0)),
        )

        subgroup_size = int(attrs.get(cuda.device_attribute.WARP_SIZE, 0))
        max_shared_memory = int(
            attrs.get(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK, 0)
        )

        try:
            bus_id = str(dev.pci_bus_id())
        except Exception:
            bus_id = f"cuda-device-{index}"

        uuid_bytes = hashlib.md5(bus_id.encode("utf-8")).digest()

        devices.append(
            (
                0,  # Vulkan variant
                int(cc_major),  # major
                int(cc_minor),  # minor
                0,  # patch
                driver_version,
                0,  # vendor id unknown in this API layer
                index,  # device id
                2,  # discrete gpu
                str(dev.name()),
                1,  # shader_buffer_float32_atomics
                1,  # shader_buffer_float32_atomic_add
                1,  # float64 support
                1 if (cc_major > 5 or (cc_major == 5 and cc_minor >= 3)) else 0,  # float16 support
                1,  # int64
                1,  # int16
                1,  # storage_buffer_16_bit_access
                1,  # uniform_and_storage_buffer_16_bit_access
                1,  # storage_push_constant_16
                1,  # storage_input_output_16
                max_workgroup_size,
                int(attrs.get(cuda.device_attribute.MAX_THREADS_PER_BLOCK, 0)),
                max_workgroup_count,
                8,  # max descriptor sets (virtualized for parity)
                4096,  # max push constant size
                min(total_memory, (1 << 31) - 1),
                65536,
                16,
                subgroup_size,
                0x7FFFFFFF,  # supported stages (virtualized for parity)
                0x7FFFFFFF,  # supported operations (virtualized for parity)
                1,
                max_shared_memory,
                [(1, 0x002)],  # compute queue
                1,  # scalar block layout
                1,  # timeline semaphores equivalent
                uuid_bytes,
            )
        )

    return devices


def context_create(device_indicies, queue_families):
    if not state._initialized:
        init(False, state._log_level)

    try:
        device_ids = [int(x) for x in device_indicies]
    except Exception:
        _set_error("context_create expected a list of integer device indices")
        return 0

    if len(device_ids) != 1:
        _set_error("CUDA Python backend currently supports exactly one device")
        return 0

    if len(queue_families) != 1 or len(queue_families[0]) != 1:
        _set_error("CUDA Python backend currently supports exactly one queue")
        return 0

    device_index = device_ids[0]

    cuda_context = None
    context_pushed = False

    try:
        if device_index < 0 or device_index >= cuda.Device.count():
            _set_error(f"Invalid CUDA device index {device_index}")
            return 0

        dev = cuda.Device(device_index)
        cc_major, _cc_minor = dev.compute_capability()
        max_kernel_param_size = _query_max_kernel_param_size(dev.device_raw, cc_major)
        uses_primary_context = False

        if hasattr(dev, "retain_primary_context"):
            cuda_context = dev.retain_primary_context()
            uses_primary_context = True
            cuda_context.push()
        else:  # pragma: no cover - fallback for older CUDA Python
            cuda_context = dev.make_context()
        context_pushed = True
        stream = cuda.Stream()

        ctx = _Context(
            device_index=device_index,
            cuda_context=cuda_context,
            streams=[stream],
            queue_count=1,
            queue_to_device=[0],
            max_kernel_param_size=int(max_kernel_param_size),
            uses_primary_context=uses_primary_context,
            stopped=False,
        )
        handle = _new_handle(state._contexts, ctx)

        # Leave no context current after creation.
        cuda.Context.pop()
        context_pushed = False
        return handle
    except Exception as exc:
        if context_pushed:
            try:
                cuda.Context.pop()
            except Exception:
                pass

        if cuda_context is not None:
            try:
                cuda_context.detach()
            except Exception:
                pass

        _set_error(f"Failed to create CUDA Python context: {exc}")
        return 0


def context_destroy(context):
    ctx = state._contexts.pop(int(context), None)
    if ctx is None:
        return

    try:
        with _activate_context(ctx):
            for stream in ctx.streams:
                stream.synchronize()
    except Exception:
        pass

    try:
        ctx.cuda_context.detach()
    except Exception:
        pass


def context_stop_threads(context):
    ctx = state._contexts.get(int(context))
    if ctx is not None:
        ctx.stopped = True


def get_error_string():
    if state._error_string is None:
        return 0
    return state._error_string


def cuda_stream_override_begin(stream_obj):
    try:
        stack = _stream_override_stack()
        stack.append(_coerce_stream_handle(stream_obj))
    except Exception as exc:
        _set_error(f"Failed to activate external CUDA stream override: {exc}")


def cuda_stream_override_end():
    stack = _stream_override_stack()
    if len(stack) > 0:
        stack.pop()
