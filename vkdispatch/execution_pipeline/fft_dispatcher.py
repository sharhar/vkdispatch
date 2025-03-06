
from typing import Tuple
from typing import Union
from typing import List

import numpy as np

import vkdispatch as vd

from typing import Dict

__fft_plans: Dict[Tuple, vd.FFTPlan] = {}

def get_fft_plan(buffer_handle: int, shape: Tuple[int, ...], do_r2c: bool, axes: List[int], inverse: bool, normalize: bool, padding, pad_freq_domain, kernel_count) -> vd.FFTPlan:
    global __fft_plans

    fft_plan_key = (
        buffer_handle, 
        *shape,
        do_r2c, 
        axes if axes is None else tuple(axes),
        normalize,
        padding if padding is None else tuple(padding),
        pad_freq_domain,
        kernel_count)

    if fft_plan_key not in __fft_plans:
        __fft_plans[fft_plan_key] = vd.FFTPlan(
            shape, 
            do_r2c, 
            axes=axes, 
            normalize=normalize,
            padding=padding,
            pad_frequency_domain=pad_freq_domain,
            kernel_count=kernel_count)

    return __fft_plans[fft_plan_key]

def reset_fft_plans():
    global __fft_plans
    __fft_plans = {}

def execute_fft_plan(
        buffer: vd.Buffer, 
        fft_shape: tuple, 
        axes: List[int] = None, 
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None, 
        do_r2c: bool = False, 
        inverse: bool = False,
        padding: List[Tuple[int, int]] = None, 
        pad_frequency_domain: bool = False,
        normalize_inverse: bool = False,
        kernel_count: int = 0,
        kernel: vd.Buffer = None):
    if cmd_stream is None:
        cmd_stream = vd.global_cmd_stream()

    plan = get_fft_plan(buffer._handle, fft_shape, do_r2c, axes, inverse, normalize_inverse, padding, pad_frequency_domain, kernel_count)
    plan.record(cmd_stream, buffer, inverse, kernel)
    
    if isinstance(cmd_stream, vd.CommandStream):
        if cmd_stream.submit_on_record:
            cmd_stream.submit()

def fft(
        buffer: vd.Buffer, 
        axes: List[int] = None,
        padding: List[Tuple[int, int]] = None, 
        pad_frequency_domain: bool = False,
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None):
    execute_fft_plan(buffer, buffer.shape, axes, cmd_stream, False, False, padding, pad_frequency_domain)

def ifft(
        buffer: vd.Buffer, 
        axes: List[int] = None, 
        normalize: bool = False,
        padding: List[Tuple[int, int]] = None, 
        pad_frequency_domain: bool = False,
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None):
    execute_fft_plan(buffer, buffer.shape, axes, cmd_stream, False, True, padding, pad_frequency_domain, normalize)

def rfft(
        buffer: vd.Buffer, 
        axes: List[int] = None,
        padding: List[Tuple[int, int]] = None, 
        pad_frequency_domain: bool = False,
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None):
    assert buffer.shape[-1] > 2, "Buffer shape must have at least 3 elements in the last dimension"

    execute_fft_plan(buffer, buffer.shape[:-1] + (buffer.shape[-1] - 2,), axes, cmd_stream, True, False, padding, pad_frequency_domain)

def irfft(
        buffer: vd.Buffer, 
        axes: List[int] = None, 
        normalize: bool = False,
        padding: List[Tuple[int, int]] = None, 
        pad_frequency_domain: bool = False,
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None):
    assert buffer.shape[-1] > 2, "Buffer shape must have at least 3 elements in the last dimension"

    execute_fft_plan(buffer, buffer.shape[:-1] + (buffer.shape[-1] - 2,), axes, cmd_stream, True, True, padding, pad_frequency_domain, normalize)

def convolve_2d(
        buffer: vd.Buffer[vd.float32],
        kernel: vd.Buffer[vd.complex64],
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None):

    assert len(buffer.shape) == 2, "Buffer must be 2D!"
    assert len(kernel.shape) == 3, "Kernel must be 3D!"

    computed_kernel_shape = (kernel.shape[0], buffer.shape[0], buffer.shape[1] // 2)

    print(computed_kernel_shape, kernel.shape)

    assert computed_kernel_shape == kernel.shape, "Kernel shape must match buffer shape!"

    execute_fft_plan(
        buffer, 
        buffer.shape[:-1] + (buffer.shape[-1] - 2,), 
        cmd_stream=cmd_stream, 
        do_r2c=True,
        kernel_count=kernel.shape[0],
        kernel=kernel)

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

        self.write(np.ascontiguousarray(data.astype(np.complex64)).view(np.float32), index)

def asrfftbuffer(data: np.ndarray) -> RFFTBuffer:
    assert not np.issubdtype(data.dtype, np.complexfloating), "Data dtype must be scalar!"

    buffer = RFFTBuffer(data.shape)
    buffer.write_real(data)

    return buffer


