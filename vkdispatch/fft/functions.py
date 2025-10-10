import vkdispatch as vd

from .shader import make_fft_shader, make_convolution_shader

from typing import Tuple, Union

def fft(
        *buffers: vd.Buffer,
        buffer_shape: Tuple = None,
        graph: vd.CommandGraph = None,
        print_shader: bool = False,
        axis: int = None,
        name: str = None,
        inverse: bool = False,
        normalize_inverse: bool = True,
        r2c: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None):
    
    assert len(buffers) >= 1, "At least one buffer must be provided"
    
    if buffer_shape is None:
        buffer_shape = buffers[0].shape

    fft_shader = make_fft_shader(
        tuple(buffer_shape),
        axis,
        inverse=inverse,
        normalize_inverse=normalize_inverse,
        r2c=r2c,
        input_map=input_map,
        output_map=output_map)

    if print_shader:
        print(fft_shader)

    fft_shader(*buffers, graph=graph)

def fft2(buffer: vd.Buffer, graph: vd.CommandGraph = None, print_shader: bool = False, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    fft(buffer, graph=graph, print_shader=print_shader, axis=len(buffer.shape) - 2, input_map=input_map)
    fft(buffer, graph=graph, print_shader=print_shader, axis=len(buffer.shape) - 1, output_map=output_map)

def fft3(buffer: vd.Buffer, graph: vd.CommandGraph = None, print_shader: bool = False, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    assert len(buffer.shape) == 3, 'Buffer must have 3 dimensions'

    fft(buffer, graph=graph, print_shader=print_shader, axis=0, input_map=input_map)
    fft(buffer, graph=graph, print_shader=print_shader, axis=1)
    fft(buffer, graph=graph, print_shader=print_shader, axis=2, output_map=output_map)


def ifft(
        buffer: vd.Buffer,
        graph: vd.CommandGraph = None,
        print_shader: bool = False,
        axis: int = None,
        name: str = None,
        normalize: bool = True,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None):
    fft(buffer, graph=graph, print_shader=print_shader, axis=axis, name=name, inverse=True, normalize_inverse=normalize, input_map=input_map, output_map=output_map)

def ifft2(buffer: vd.Buffer, graph: vd.CommandGraph = None, print_shader: bool = False, normalize: bool = True, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    ifft(buffer, graph=graph, print_shader=print_shader, axis=len(buffer.shape) - 2, normalize=normalize, input_map=input_map)
    ifft(buffer, graph=graph, print_shader=print_shader, axis=len(buffer.shape) - 1, normalize=normalize, output_map=output_map)

def ifft3(buffer: vd.Buffer, graph: vd.CommandGraph = None, print_shader: bool = False, normalize: bool = True, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    assert len(buffer.shape) == 3, 'Buffer must have 3 dimensions'

    ifft(buffer, graph=graph, print_shader=print_shader, axis=0, normalize=normalize, input_map=input_map)
    ifft(buffer, graph=graph, print_shader=print_shader, axis=1, normalize=normalize)
    ifft(buffer, graph=graph, print_shader=print_shader, axis=2, normalize=normalize, output_map=output_map)


def rfft(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False, name: str = None):
    fft(buffer, buffer_shape=buffer.real_shape, graph=graph, print_shader=print_shader, name=name, r2c=True)

def rfft2(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False):
    assert len(buffer.real_shape) == 2 or len(buffer.real_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    rfft(buffer, graph=graph, print_shader=print_shader)
    fft(buffer, graph=graph, print_shader=print_shader, axis=len(buffer.real_shape) - 2)

def rfft3(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False):
    assert len(buffer.real_shape) == 3, 'Buffer must have 3 dimensions'

    rfft(buffer, graph=graph, print_shader=print_shader)
    fft(buffer, graph=graph, print_shader=print_shader, axis=1)
    fft(buffer, graph=graph, print_shader=print_shader, axis=0)

def irfft(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False, name: str = None, normalize: bool = True):
    fft(buffer, buffer_shape=buffer.real_shape, graph=graph, print_shader=print_shader, name=name, inverse=True, normalize_inverse=normalize, r2c=True)

def irfft2(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False, normalize: bool = True):
    assert len(buffer.real_shape) == 2 or len(buffer.real_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    ifft(buffer, graph=graph, print_shader=print_shader, axis=len(buffer.real_shape) - 2, normalize=normalize)
    irfft(buffer, graph=graph, print_shader=print_shader, normalize=normalize)

def irfft3(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False, normalize: bool = True):
    assert len(buffer.real_shape) == 3, 'Buffer must have 3 dimensions'

    ifft(buffer, graph=graph, print_shader=print_shader, axis=0, normalize=normalize)
    ifft(buffer, graph=graph, print_shader=print_shader, axis=1, normalize=normalize)
    irfft(buffer, graph=graph, print_shader=print_shader, normalize=normalize)

def convolve(
        *buffers: vd.Buffer,
        kernel_map: vd.MappingFunction = None,
        kernel_num: int = 1,
        buffer_shape: Tuple = None,
        graph: vd.CommandGraph = None,
        print_shader: bool = False,
        axis: int = None,
        normalize: bool = True,
        name: str = None,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None):
    if buffer_shape is None:
        buffer_shape = buffers[0].shape

    fft_shader = make_convolution_shader(
        tuple(buffer_shape),
        kernel_map,
        kernel_num,
        axis,
        normalize=normalize,
        input_map=input_map,
        output_map=output_map)

    if print_shader:
        print(fft_shader)

    fft_shader(*buffers, graph=graph)

def convolve2D(
        buffer: vd.Buffer,
        kernel: vd.Buffer,
        kernel_map: vd.MappingFunction = None,
        buffer_shape: Tuple = None,
        graph: vd.CommandGraph = None,
        print_shader: bool = False,
        normalize: bool = True,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None):

    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    input_buffers = [buffer]

    if input_map is not None:
        input_buffers.append(buffer)

    output_buffers = [buffer]
    if output_map is not None:
        output_buffers.append(buffer)

    fft(*input_buffers, graph=graph, print_shader=print_shader, input_map=input_map)
    convolve(buffer, kernel, kernel_map=kernel_map, buffer_shape=buffer_shape, graph=graph, print_shader=print_shader, axis=len(buffer.shape) - 2, normalize=normalize)
    ifft(*output_buffers, graph=graph, print_shader=print_shader, normalize=normalize, output_map=output_map)

def convolve2DR(
        buffer: vd.RFFTBuffer,
        kernel: vd.RFFTBuffer,
        kernel_map: vd.MappingFunction = None,
        buffer_shape: Tuple = None,
        graph: vd.CommandGraph = None,
        print_shader: bool = False,
        normalize: bool = True):
    
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    rfft(buffer, graph=graph, print_shader=print_shader)
    convolve(buffer, kernel, kernel_map=kernel_map, buffer_shape=buffer_shape, graph=graph, print_shader=print_shader, axis=len(buffer.shape) - 2, normalize=normalize)
    irfft(buffer, graph=graph, print_shader=print_shader, normalize=normalize)