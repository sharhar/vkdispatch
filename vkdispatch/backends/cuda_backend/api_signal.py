from __future__ import annotations

from . import state as state
from .helpers import (
    activate_context,
    context_from_handle,
    new_handle,
    query_signal,
    queue_indices,
    record_signal,
    set_error,
    stream_for_queue,
)
from .state import CUDASignal


def signal_wait(signal_ptr, wait_for_timestamp, queue_index):
    signal_obj = state.signals.get(int(signal_ptr))
    if signal_obj is None:
        return True

    if not bool(wait_for_timestamp):
        # CUDA Python records signals synchronously on submission; host-side "recorded" waits
        # should therefore complete immediately once an event exists.
        if signal_obj.event is None:
            return bool(signal_obj.done)
        return bool(signal_obj.submitted)

    if signal_obj.done:
        return True

    if signal_obj.event is None:
        return bool(signal_obj.done)

    ctx = state.contexts.get(signal_obj.context_handle)
    if ctx is None:
        return query_signal(signal_obj)

    try:
        with activate_context(ctx):
            signal_obj.event.synchronize()
        signal_obj.done = True
        return True
    except Exception:
        return query_signal(signal_obj)


def signal_insert(context, queue_index):
    ctx = context_from_handle(int(context))
    if ctx is None:
        return 0

    selected = queue_indices(ctx, int(queue_index))
    if len(selected) == 0:
        selected = [0]

    signal = CUDASignal(context_handle=int(context), queue_index=selected[0], submitted=False, done=False)
    handle = new_handle(state.signals, signal)

    try:
        with activate_context(ctx):
            record_signal(signal, stream_for_queue(ctx, selected[0]))
    except Exception as exc:
        set_error(f"Failed to insert signal: {exc}")
        return 0

    return handle


def signal_destroy(signal_ptr):
    state.signals.pop(int(signal_ptr), None)
