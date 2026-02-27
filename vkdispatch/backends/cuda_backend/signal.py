from __future__ import annotations

from . import state as state
from .helpers import (
    activate_context,
    context_from_handle,
    new_handle,
    queue_indices,
    set_error,
    stream_for_queue,
)

from typing import Optional, Dict

from .cuda_primitives import cuda
from .handle import CUDAHandle, HandleRegistry

_signals: HandleRegistry = HandleRegistry()

class CUDASignal(CUDAHandle):
    context_handle: int
    queue_index: int
    event: Optional["cuda.Event"] = None
    submitted: bool = True
    done: bool = True

    def __init__(self,
                context_handle: int,
                queue_index: int,
                event: Optional["cuda.Event"] = None,
                submitted: bool = True,
                done: bool = True):
        super().__init__(_signals)

        self.context_handle = context_handle
        self.queue_index = queue_index
        self.event = event
        self.submitted = submitted
        self.done = done

    @staticmethod
    def from_handle(handle: int) -> Optional["CUDASignal"]:
        return _signals.get(handle)

    def record(self, stream: "cuda.Stream"):
        self.submitted = True
        self.done = False
        if self.event is None:
            self.event = cuda.Event()
        self.event.record(stream)

    def query(self) -> bool:
        if self.event is None:
            return bool(self.done)

        try:
            done = self.event.query()
        except Exception:
            return False

        self.done = bool(done)
        return self.done

def signal_wait(signal_ptr, wait_for_timestamp, queue_index):
    signal_obj = CUDASignal.from_handle(signal_ptr)
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
        return signal_obj.query()

    try:
        with activate_context(ctx):
            signal_obj.event.synchronize()
        signal_obj.done = True
        return True
    except Exception:
        return signal_obj.query()


def signal_insert(context, queue_index):
    ctx = context_from_handle(int(context))
    if ctx is None:
        return 0

    selected = queue_indices(ctx, int(queue_index))
    if len(selected) == 0:
        selected = [0]

    signal = CUDASignal(context_handle=int(context), queue_index=selected[0], submitted=False, done=False)

    try:
        with activate_context(ctx):
            signal.record(stream_for_queue(ctx, selected[0]))
    except Exception as exc:
        set_error(f"Failed to insert signal: {exc}")
        return 0

    return signal.handle


def signal_destroy(signal_ptr):
    _signals.pop(signal_ptr)
