
from typing import Tuple
from typing import Union
from typing import List

import numpy as np

import vkdispatch as vd

from .fft_plan import VkFFTPlan

import dataclasses

from typing import Dict
from typing import Union

@dataclasses.dataclass(frozen=True)
class FFTConfig:
    buffer_handle: int
    shape: Tuple[int, ...]
    do_r2c: bool = False
    axes: Tuple[int] = None
    normalize: bool = False
    padding: Tuple[Tuple[int, int]] = None
    pad_freq_domain: bool = False
    kernel_count: int = 0
    input_shape: Tuple[int, ...] = None
    input_type: vd.dtype = None
    kernel_convolution: bool = False
    conjugate_convolution: bool = False
    convolution_features: int = 1
    num_batches: int = 1
    single_kernel_multiple_batches: bool = False
    keep_shader_code: bool = False

def sanitize_input_tuple(input: Tuple) -> Tuple:
    if input is None:
        return None

    return tuple(input)

__fft_plans: Dict[FFTConfig, VkFFTPlan] = {}

def reset_fft_plans():
    global __fft_plans
    __fft_plans = {}

def execute_fft_plan(
        buffer: vd.Buffer,
        inverse: bool,
        config: FFTConfig,
        kernel: vd.Buffer = None,
        input: vd.Buffer = None,
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None):
    if cmd_stream is None:
        cmd_stream = vd.global_cmd_stream()
    
    if config not in __fft_plans:
        __fft_plans[config] = VkFFTPlan(
            shape=config.shape, 
            do_r2c=config.do_r2c, 
            axes=config.axes, 
            normalize=config.normalize, 
            padding=config.padding, 
            pad_frequency_domain=config.pad_freq_domain, 
            kernel_count=config.kernel_count,
            input_shape=config.input_shape,
            input_type=config.input_type,
            kernel_convolution=config.kernel_convolution,
            conjugate_convolution=config.conjugate_convolution,
            convolution_features=config.convolution_features,
            num_batches=config.num_batches,
            keep_shader_code=config.keep_shader_code
        )
    
    plan = __fft_plans[config]
    plan.record(cmd_stream, buffer, inverse, kernel, input)

    if isinstance(cmd_stream, vd.CommandStream):
        if cmd_stream.submit_on_record:
            cmd_stream.submit()

def fft_old(
        buffer: vd.Buffer,
        input: vd.Buffer = None,
        axes: List[int] = None,
        padding: List[Tuple[int, int]] = None, 
        pad_frequency_domain: bool = False,
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None,
        keep_shader_code: bool = False):
    
    execute_fft_plan(
        buffer,
        False,
        cmd_stream = cmd_stream,
        config = FFTConfig(
            buffer_handle=buffer._handle,
            shape=sanitize_input_tuple(buffer.shape),
            axes=sanitize_input_tuple(axes),
            padding=sanitize_input_tuple(padding),
            pad_freq_domain=pad_frequency_domain,
            input_shape=sanitize_input_tuple(input.shape if input is not None else None),
            input_type=input.var_type if input is not None else None,
            keep_shader_code=keep_shader_code
        ),
        input=input
    )

def ifft_old(
        buffer: vd.Buffer, 
        input: vd.Buffer = None,
        axes: List[int] = None, 
        normalize: bool = False,
        padding: List[Tuple[int, int]] = None, 
        pad_frequency_domain: bool = False,
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None,\
        keep_shader_code: bool = False):

    execute_fft_plan(
        buffer,
        True,
        cmd_stream = cmd_stream,
        config = FFTConfig(
            buffer_handle=buffer._handle,
            shape=sanitize_input_tuple(buffer.shape),
            axes=sanitize_input_tuple(axes),
            normalize=normalize,
            padding=sanitize_input_tuple(padding),
            pad_freq_domain=pad_frequency_domain,
            input_shape=sanitize_input_tuple(input.shape if input is not None else None),
            input_type=input.var_type if input is not None else None,
            keep_shader_code=keep_shader_code
        ),
        input=input
        )

def rfft_old(
        buffer: vd.RFFTBuffer,
        input: vd.Buffer = None,
        axes: List[int] = None,
        padding: List[Tuple[int, int]] = None, 
        pad_frequency_domain: bool = False,
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None,
        keep_shader_code: bool = False):
    assert buffer.shape[-1] > 2, "Buffer shape must have at least 3 elements in the last dimension"

    execute_fft_plan(
        buffer,
        False,
        cmd_stream = cmd_stream,
        config = FFTConfig(
            buffer_handle=buffer._handle,
            shape=sanitize_input_tuple(buffer.real_shape),
            do_r2c=True,
            axes=sanitize_input_tuple(axes),
            padding=sanitize_input_tuple(padding),
            pad_freq_domain=pad_frequency_domain,
            input_shape=sanitize_input_tuple(input.shape if input is not None else None),
            input_type=input.var_type if input is not None else None,
            keep_shader_code=keep_shader_code
        ),
        input=input
    )

def irfft_old(
        buffer: vd.RFFTBuffer,
        input: vd.Buffer = None,
        axes: List[int] = None, 
        normalize: bool = False,
        padding: List[Tuple[int, int]] = None, 
        pad_frequency_domain: bool = False,
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None,
        keep_shader_code: bool = False):
    assert buffer.shape[-1] > 2, "Buffer shape must have at least 3 elements in the last dimension"

    execute_fft_plan(
        buffer,
        True,
        cmd_stream = cmd_stream,
        config = FFTConfig(
            buffer_handle=buffer._handle,
            shape=sanitize_input_tuple(buffer.real_shape),
            do_r2c=True,
            axes=sanitize_input_tuple(axes),
            normalize=normalize,
            padding=sanitize_input_tuple(padding),
            pad_freq_domain=pad_frequency_domain,
            input_shape=sanitize_input_tuple(input.shape if input is not None else None),
            input_type=input.var_type if input is not None else None,
            keep_shader_code=keep_shader_code
        ),
        input=input
    )

def sanitize_2d_convolution_buffer_shape(in_shape: vd.Buffer):
    if in_shape is None:
        return None
    
    in_shape = in_shape.shape

    assert len(in_shape) == 2 or len(in_shape) == 3, "Input shape must be 2D or 3D!"

    if len(in_shape) == 2:
        return (1, in_shape[0], in_shape[1])
    
    return in_shape

def convolve_2Dreal(
        buffer: vd.RFFTBuffer,
        kernel: Union[vd.Buffer[vd.float32], vd.RFFTBuffer],
        input: Union[vd.Buffer[vd.float32], vd.RFFTBuffer] = None,
        normalize: bool = False,
        conjugate_kernel: bool = False,
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None,
        keep_shader_code: bool = False):

    buffer_shape = sanitize_2d_convolution_buffer_shape(buffer)
    kernel_shape = sanitize_2d_convolution_buffer_shape(kernel)
    
    assert buffer_shape == kernel_shape, f"Buffer ({buffer_shape}) and Kernel ({kernel_shape}) shapes must match!"

    input_shape = sanitize_2d_convolution_buffer_shape(input)

    kernel_count = 1
    feature_count = 1

    if input_shape is not None:
        assert buffer_shape[0] % input_shape[0] == 0, f"Output count ({buffer_shape[0]}) must be divisible by input count ({input_shape[0]})!"
        kernel_count = buffer_shape[0] // input_shape[0]
        feature_count = input_shape[0]
    else:
        feature_count = buffer.shape[0]

    execute_fft_plan(
        buffer,
        False,
        cmd_stream = cmd_stream,
        config = FFTConfig(
            buffer_handle=buffer._handle,
            shape=sanitize_input_tuple(buffer.real_shape),
            do_r2c=True,
            normalize=normalize,
            kernel_count=kernel_count,
            conjugate_convolution=conjugate_kernel,
            input_shape=sanitize_input_tuple(input.shape if input is not None else None),
            input_type=input.var_type if input is not None else None,
            convolution_features=feature_count,
            keep_shader_code=keep_shader_code
        ),
        kernel=kernel,
        input=input
    )

def create_kernel_2Dreal(
        kernel: vd.RFFTBuffer,
        shape: Tuple[int, ...] = None,
        feature_count: int = 1,
        cmd_stream: Union[vd.CommandList, vd.CommandStream, None] = None,
        keep_shader_code: bool = False) -> vd.RFFTBuffer:
    
    if shape is None:
        shape = kernel.shape

    if len(shape) == 2:
        assert feature_count == 1, "Feature count must be 1 for 2D kernels!"
        shape = (1,) + shape

    execute_fft_plan(
        kernel,
        False,
        cmd_stream = cmd_stream,
        config = FFTConfig(
            buffer_handle=kernel._handle,
            shape=sanitize_input_tuple(kernel.real_shape),
            do_r2c=True,
            kernel_convolution=True,
            convolution_features=feature_count,
            num_batches=shape[0] // feature_count,
            keep_shader_code=keep_shader_code
        )
    )

    return kernel


def fft(
        buffer: vd.Buffer,
        input_buffer: vd.Buffer = None,
        buffer_shape: Tuple = None,
        cmd_stream: vd.CommandStream = None,
        print_shader: bool = False,
        axis: int = None,
        inverse: bool = False,
        normalize_inverse: bool = True,
        r2c: bool = False):
    
    if axis is None:
        axis = [len(buffer.shape) - 1]
    
    if isinstance(axis, int):
        axis = [axis]

    if buffer_shape is None:
        buffer_shape = buffer.shape

    config = FFTConfig(
        buffer_handle=buffer._handle,
        shape=sanitize_input_tuple(buffer_shape),
        axes=sanitize_input_tuple(axis),
        padding=None,
        pad_freq_domain=False,
        input_shape=sanitize_input_tuple(input_buffer.shape if input_buffer is not None else None),
        input_type=input_buffer.var_type if input_buffer is not None else None,
        keep_shader_code=print_shader,
        normalize=normalize_inverse,
        do_r2c=r2c
    )

    execute_fft_plan(
        buffer,
        inverse=inverse,
        cmd_stream=cmd_stream,
        input=input_buffer,
        config=config
    )