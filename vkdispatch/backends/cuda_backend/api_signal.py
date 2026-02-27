from __future__ import annotations

from . import _state as state
from ._helpers import (
    _activate_context,
    _context_from_handle,
    _new_handle,
    _query_signal,
    _queue_indices,
    _record_signal,
    _set_error,
    _stream_for_queue,
)
from ._state import _Signal


def signal_wait(signal_ptr, wait_for_timestamp, queue_index):
    signal_obj = state._signals.get(int(signal_ptr))
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

    ctx = state._contexts.get(signal_obj.context_handle)
    if ctx is None:
        return _query_signal(signal_obj)

    try:
        with _activate_context(ctx):
            signal_obj.event.synchronize()
        signal_obj.done = True
        return True
    except Exception:
        return _query_signal(signal_obj)


def signal_insert(context, queue_index):
    ctx = _context_from_handle(int(context))
    if ctx is None:
        return 0

    selected = _queue_indices(ctx, int(queue_index))
    if len(selected) == 0:
        selected = [0]

    signal = _Signal(context_handle=int(context), queue_index=selected[0], submitted=False, done=False)
    handle = _new_handle(state._signals, signal)

    try:
        with _activate_context(ctx):
            _record_signal(signal, _stream_for_queue(ctx, selected[0]))
    except Exception as exc:
        _set_error(f"Failed to insert signal: {exc}")
        return 0

    return handle


def signal_destroy(signal_ptr):
    state._signals.pop(int(signal_ptr), None)
