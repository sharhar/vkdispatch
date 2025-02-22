
from typing import Tuple
from typing import Union
from typing import List

import numpy as np

import vkdispatch as vd

__fft_plans = {}

def get_fft_plan(buffer_handle: int, shape: Tuple[int, ...], do_r2c: bool) -> vd.FFTPlan:
    global __fft_plans

    fft_plan_key = (buffer_handle, *shape, do_r2c)

    if fft_plan_key not in __fft_plans:
        __fft_plans[fft_plan_key] = vd.FFTPlan(shape, do_r2c)

    return __fft_plans[fft_plan_key]


def reset_fft_plans():
    global __fft_plans
    __fft_plans = {}

def execute_fft_plan(buffer: vd.Buffer, fft_shape: tuple, cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None, do_r2c: bool = False, inverse: bool = False):
    if cmd_stream is None:
        cmd_stream = vd.global_cmd_stream()

    plan = get_fft_plan(buffer._handle, fft_shape, do_r2c)
    plan.record(cmd_stream, buffer, inverse)
    
    if isinstance(cmd_stream, vd.CommandStream):
        if cmd_stream.submit_on_record:
            cmd_stream.submit()

def fft(buffer: vd.Buffer, cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None):
    execute_fft_plan(buffer, buffer.shape, cmd_stream, False, False)

def ifft(buffer: vd.Buffer, cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None):
    execute_fft_plan(buffer, buffer.shape, cmd_stream, False, True)

def rfft(buffer: vd.Buffer, cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None):
    assert buffer.shape[-1] > 2, "Buffer shape must have at least 3 elements in the last dimension"

    execute_fft_plan(buffer, buffer.shape[:-1] + (buffer.shape[-1] - 2,), cmd_stream, True, False)

def irfft(buffer: vd.Buffer, cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None):
    assert buffer.shape[-1] > 2, "Buffer shape must have at least 3 elements in the last dimension"

    execute_fft_plan(buffer, buffer.shape[:-1] + (buffer.shape[-1] - 2,), cmd_stream, True, True)

class RFFTBuffer(vd.Buffer):
    def __init__(self, shape: Tuple[int, ...]):
        assert shape[-1] % 2 == 0, "Last dimension of RFFTBuffer must be even!"
        super().__init__(shape[:-1] + (shape[-1] + 2,), vd.float32)

        self.real_shape = shape
        self.fourier_shape = self.shape[:-1] + (self.shape[-1] / 2,)
    
    def read_real(self, index: Union[int, None] = None) -> np.ndarray:
        return self.read(index)[..., :self.real_shape[-1]]

    def read_fourier(self, index: Union[int, None] = None) -> np.ndarray:
        return self.read(index).view(np.complex64)
    
    def write_real(self, data: np.ndarray, index: int = -1):
        assert data.shape == self.real_shape, "Data shape must match real shape!"
        assert not np.issubdtype(data.dtype, np.complexfloating) , "Data dtype must be scalar!"

        true_data = np.zeros(self.shape, dtype=np.float32)
        true_data[..., :self.real_shape[-1]] = data

        self.write(true_data, index)

    def write_fourier(self, data: np.ndarray, index: int = -1):
        assert data.shape == self.fourier_shape, "Data shape must match fourier shape!"
        assert np.issubdtype(data.dtype, np.complexfloating) , "Data dtype must be complex!"

        self.write(data.astype(np.complex64).view(np.float32), index)


def asrfftbuffer(data: np.ndarray) -> RFFTBuffer:
    assert not np.issubdtype(data.dtype, np.complexfloating), "Data dtype must be scalar!"

    buffer = RFFTBuffer(data.shape)
    buffer.write_real(data)

    return buffer


