from __future__ import annotations

from . import state as state
from .cuda_primitives import cuda
from .helpers import (
    activate_context,
    allocate_staging_storage,
    buffer_device_ptr,
    context_from_handle,
    new_handle,
    query_signal,
    queue_indices,
    record_signal,
    set_error,
    stream_for_queue,
    to_bytes,
)
from .state import CUDABuffer, CUDASignal


def buffer_create(context, size, per_device):
    _ = per_device

    ctx = context_from_handle(int(context))
    if ctx is None:
        return 0

    size = int(size)
    if size <= 0:
        set_error("Buffer size must be greater than zero")
        return 0

    try:
        with activate_context(ctx):
            allocation = cuda.mem_alloc(size)

        signal_handles = [
            new_handle(state.signals, CUDASignal(context_handle=int(context), queue_index=i, done=True))
            for i in range(ctx.queue_count)
        ]

        obj = CUDABuffer(
            context_handle=int(context),
            size=size,
            device_ptr=int(allocation),
            device_allocation=allocation,
            owns_allocation=True,
            staging_data=[allocate_staging_storage(size) for _ in range(ctx.queue_count)],
            signal_handles=signal_handles,
        )
        return new_handle(state.buffers, obj)
    except Exception as exc:
        set_error(f"Failed to create CUDA buffer: {exc}")
        return 0


def buffer_create_external(context, size, device_ptr):
    ctx = context_from_handle(int(context))
    if ctx is None:
        return 0

    size = int(size)
    device_ptr = int(device_ptr)

    if size <= 0:
        set_error("External buffer size must be greater than zero")
        return 0

    if device_ptr == 0:
        set_error("External buffer device pointer must be non-zero")
        return 0

    try:
        signal_handles = [
            new_handle(state.signals, CUDASignal(context_handle=int(context), queue_index=i, done=True))
            for i in range(ctx.queue_count)
        ]

        obj = CUDABuffer(
            context_handle=int(context),
            size=size,
            device_ptr=device_ptr,
            device_allocation=None,
            owns_allocation=False,
            staging_data=[allocate_staging_storage(size) for _ in range(ctx.queue_count)],
            signal_handles=signal_handles,
        )
        return new_handle(state.buffers, obj)
    except Exception as exc:
        set_error(f"Failed to create external CUDA buffer alias: {exc}")
        return 0


def buffer_destroy(buffer):
    obj = state.buffers.pop(int(buffer), None)
    if obj is None:
        return

    for signal_handle in obj.signal_handles:
        state.signals.pop(signal_handle, None)

    ctx = state.contexts.get(obj.context_handle)
    if ctx is None or not obj.owns_allocation or obj.device_allocation is None:
        return

    try:
        with activate_context(ctx):
            obj.device_allocation.free()
    except Exception:
        pass


def buffer_get_queue_signal(buffer, queue_index):
    obj = state.buffers.get(int(buffer))
    if obj is None:
        return new_handle(state.signals, CUDASignal(context_handle=0, queue_index=0, done=True))

    queue_index = int(queue_index)
    if queue_index < 0 or queue_index >= len(obj.signal_handles):
        queue_index = 0

    return obj.signal_handles[queue_index]


def buffer_wait_staging_idle(buffer, queue_index):
    signal_handle = buffer_get_queue_signal(buffer, queue_index)
    signal_obj = state.signals.get(int(signal_handle))
    if signal_obj is None:
        return True
    return query_signal(signal_obj)


def buffer_write_staging(buffer, queue_index, data, size):
    obj = state.buffers.get(int(buffer))
    if obj is None:
        return

    queue_index = int(queue_index)
    if queue_index < 0 or queue_index >= len(obj.staging_data):
        return

    payload = to_bytes(data)
    size = min(int(size), len(payload), obj.size)
    if size <= 0:
        return

    payload_view = memoryview(payload)[:size]
    staging_view = memoryview(obj.staging_data[queue_index])
    staging_view[:size] = payload_view


def buffer_read_staging(buffer, queue_index, size):
    obj = state.buffers.get(int(buffer))
    if obj is None:
        return bytes(int(size))

    queue_index = int(queue_index)
    if queue_index < 0 or queue_index >= len(obj.staging_data):
        return bytes(int(size))

    size = max(0, int(size))
    staging = obj.staging_data[queue_index]

    if size <= len(staging):
        return bytes(staging[:size])

    return bytes(staging) + bytes(size - len(staging))


def buffer_write(buffer, offset, size, index):
    obj = state.buffers.get(int(buffer))
    if obj is None:
        return

    ctx = state.contexts.get(obj.context_handle)
    if ctx is None:
        set_error(f"Missing context for buffer handle {buffer}")
        return

    offset = int(offset)
    size = int(size)
    if size <= 0 or offset < 0:
        return

    try:
        with activate_context(ctx):
            for queue_index in queue_indices(ctx, int(index), all_on_negative=True):
                stream = stream_for_queue(ctx, queue_index)
                end = min(offset + size, obj.size)
                copy_size = end - offset
                if copy_size <= 0:
                    continue

                src_view = memoryview(obj.staging_data[queue_index])[:copy_size]
                cuda.memcpy_htod_async(buffer_device_ptr(obj) + offset, src_view, stream)

                signal = state.signals.get(obj.signal_handles[queue_index])
                if signal is not None:
                    record_signal(signal, stream)
    except Exception as exc:
        set_error(f"Failed to write CUDA buffer: {exc}")


def buffer_read(buffer, offset, size, index):
    obj = state.buffers.get(int(buffer))
    if obj is None:
        return

    ctx = state.contexts.get(obj.context_handle)
    if ctx is None:
        set_error(f"Missing context for buffer handle {buffer}")
        return

    queue_index = int(index)
    if queue_index < 0 or queue_index >= ctx.queue_count:
        set_error(f"Invalid queue index {queue_index} for buffer read")
        return

    offset = int(offset)
    size = int(size)
    if size <= 0 or offset < 0:
        return

    try:
        with activate_context(ctx):
            stream = stream_for_queue(ctx, queue_index)
            end = min(offset + size, obj.size)
            copy_size = end - offset
            if copy_size <= 0:
                return

            dst_view = memoryview(obj.staging_data[queue_index])[:copy_size]
            cuda.memcpy_dtoh_async(dst_view, buffer_device_ptr(obj) + offset, stream)

            signal = state.signals.get(obj.signal_handles[queue_index])
            if signal is not None:
                record_signal(signal, stream)
    except Exception as exc:
        set_error(f"Failed to read CUDA buffer: {exc}")
