
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

def clear_plan_cache():
    global __fft_plans

    for plan in __fft_plans.values():
        plan.destroy()

    __fft_plans = {}

def execute_fft_plan(
        buffer: vd.Buffer,
        inverse: bool,
        config: FFTConfig,
        kernel: vd.Buffer = None,
        input: vd.Buffer = None,
        graph: Union[vd.CommandList, vd.CommandGraph, None] = None):
    if graph is None:
        graph = vd.global_graph()
    
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
    plan.record(graph, buffer, inverse, kernel, input)

    if isinstance(graph, vd.CommandGraph):
        if graph.submit_on_record:
            graph.submit()

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
        graph: Union[vd.CommandList, vd.CommandGraph, None] = None,
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
        graph = graph,
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
        graph: Union[vd.CommandList, vd.CommandGraph, None] = None,
        keep_shader_code: bool = False) -> vd.RFFTBuffer:
    
    if shape is None:
        shape = kernel.shape

    if len(shape) == 2:
        assert feature_count == 1, "Feature count must be 1 for 2D kernels!"
        shape = (1,) + shape

    execute_fft_plan(
        kernel,
        False,
        graph = graph,
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


def convolve_2D(
        buffer: vd.Buffer,
        kernel: Union[vd.Buffer[vd.float32], vd.Buffer],
        normalize: bool = False,
        conjugate_kernel: bool = False,
        graph: Union[vd.CommandList, vd.CommandGraph, None] = None,
        keep_shader_code: bool = False,
        padding: Tuple[Tuple[int, int]] = None):

    buffer_shape = sanitize_2d_convolution_buffer_shape(buffer)
    kernel_shape = sanitize_2d_convolution_buffer_shape(kernel)
    
    # assert buffer_shape == kernel_shape, f"Buffer ({buffer_shape}) and Kernel ({kernel_shape}) shapes must match!"

    kernel_count = kernel.shape[0] if len(kernel.shape) == 3 else 1
    feature_count = 1

    if kernel_count > 1:
        feature_count = buffer.shape[0]

    in_shape = sanitize_input_tuple(buffer.shape)

    execute_fft_plan(
        buffer,
        False,
        graph = graph,
        config = FFTConfig(
            buffer_handle=buffer._handle,
            shape=in_shape[1:], # if kernel_count == 1 else in_shape,
            normalize=normalize,
            kernel_count=1, #kernel_count,
            conjugate_convolution=conjugate_kernel,
            convolution_features=1, #feature_count,
            keep_shader_code=keep_shader_code,
            num_batches=buffer.shape[0], # if kernel_count == 1 else 1,
            padding=padding
        ),
        kernel=kernel
    )

def fft(
        buffer: vd.Buffer,
        input_buffer: vd.Buffer = None,
        buffer_shape: Tuple = None,
        graph: vd.CommandGraph = None,
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
        graph=graph,
        input=input_buffer,
        config=config
    )

def fft2(buffer: vd.Buffer, graph: vd.CommandGraph = None, print_shader: bool = False):
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    axes = (len(buffer.shape) - 2, len(buffer.shape) - 1)

    fft(buffer, graph=graph, print_shader=print_shader, axis=axes)

def fft3(buffer: vd.Buffer, graph: vd.CommandGraph = None, print_shader: bool = False):
    assert len(buffer.shape) == 3, 'Buffer must have 3 dimensions'

    fft(buffer, graph=graph, print_shader=print_shader, axis=(0, 1, 2))

def ifft(
        buffer: vd.Buffer,
        graph: vd.CommandGraph = None,
        print_shader: bool = False,
        axis: int = None,
        normalize: bool = True):
    fft(buffer, graph=graph, print_shader=print_shader, axis=axis, inverse=True, normalize_inverse=normalize)

def ifft2(buffer: vd.Buffer, graph: vd.CommandGraph = None, print_shader: bool = False):
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    axes = (len(buffer.shape) - 2, len(buffer.shape) - 1)

    ifft(buffer, graph=graph, print_shader=print_shader, axis=axes)

def ifft3(buffer: vd.Buffer, graph: vd.CommandGraph = None, print_shader: bool = False):
    assert len(buffer.shape) == 3, 'Buffer must have 3 dimensions'

    ifft(buffer, graph=graph, print_shader=print_shader, axis=(0, 1, 2))


def rfft(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False, axis: int = None):
    fft(buffer, buffer_shape=buffer.real_shape, graph=graph, print_shader=print_shader, r2c=True, axis=axis)

def rfft2(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False):
    assert len(buffer.real_shape) == 2 or len(buffer.real_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    axes = (len(buffer.shape) - 2, len(buffer.shape) - 1)
    rfft(buffer, graph=graph, print_shader=print_shader, axis=axes)

def rfft3(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False):
    assert len(buffer.real_shape) == 3, 'Buffer must have 3 dimensions'

    rfft(buffer, graph=graph, print_shader=print_shader, axis=(0, 1, 2))

def irfft(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False, normalize: bool = True, axis: int = None):
    fft(buffer, buffer_shape=buffer.real_shape, graph=graph, print_shader=print_shader, inverse=True, normalize_inverse=normalize, r2c=True, axis=axis)

def irfft2(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False):
    assert len(buffer.real_shape) == 2 or len(buffer.real_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    axes = (len(buffer.shape) - 2, len(buffer.shape) - 1)
    irfft(buffer, graph=graph, print_shader=print_shader, axis=axes)

def irfft3(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False):
    assert len(buffer.real_shape) == 3, 'Buffer must have 3 dimensions'

    irfft(buffer, graph=graph, print_shader=print_shader, axis=(0, 1, 2))