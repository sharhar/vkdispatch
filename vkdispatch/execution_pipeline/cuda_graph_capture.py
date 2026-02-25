import vkdispatch as vd

from contextlib import contextmanager

import threading

import typing

class CUDAGraphCapture:
    cuda_stream = typing.Any
    uniform_buffers = typing.List[typing.Any]

    def add_uniform_buffer(self, buffer):
        self.uniform_buffers.append(buffer)

_cap = threading.local()

def _set_capture(capture):
    _cap.capture = capture

def get_cuda_capture() -> CUDAGraphCapture:
    return getattr(_cap, "capture", None)

@contextmanager
def cuda_graph_capture(cuda_stream=None):
    assert vd.is_cuda(), "CUDA graph capture is only supported when using the CUDA backend."

    cap = CUDAGraphCapture()
    cap.cuda_stream = cuda_stream
    cap.uniform_buffers = []

    _set_capture(cap)
    
    try:
        yield cap
    finally:
        _set_capture(None)