import vkdispatch as vd

from .fft_shader import make_fft_shader

from typing import Tuple

def fft(
        buffer: vd.Buffer,
        buffer_shape: Tuple = None,
        cmd_stream: vd.CommandStream = None,
        print_shader: bool = False,
        axis: int = None,
        name: str = None,
        inverse: bool = False,
        normalize_inverse: bool = True,
        r2c: bool = False):
    
    if buffer_shape is None:
        buffer_shape = buffer.shape

    fft_shader, exec_size = make_fft_shader(buffer_shape, axis, name=name, inverse=inverse, normalize_inverse=normalize_inverse, r2c=r2c)

    if print_shader:
        print(fft_shader)

    fft_shader(buffer, cmd_stream=cmd_stream, exec_size=exec_size)

def fft2(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False):
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=len(buffer.shape) - 2)
    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=len(buffer.shape) - 1)

def fft3(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False):
    assert len(buffer.shape) == 3, 'Buffer must have 3 dimensions'

    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=0)
    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=1)
    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=2)


def ifft(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False, axis: int = None, name: str = None, normalize: bool = True):
    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=axis, name=name, inverse=True, normalize_inverse=normalize)

def ifft2(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False, normalize: bool = True):
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    ifft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=len(buffer.shape) - 2, normalize=normalize)
    ifft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=len(buffer.shape) - 1, normalize=normalize)

def ifft3(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False, normalize: bool = True):
    assert len(buffer.shape) == 3, 'Buffer must have 3 dimensions'

    ifft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=0, normalize=normalize)
    ifft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=1, normalize=normalize)
    ifft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=2, normalize=normalize)


def rfft(buffer: vd.RFFTBuffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False, name: str = None):
    fft(buffer, buffer_shape=buffer.real_shape, cmd_stream=cmd_stream, print_shader=print_shader, name=name, r2c=True)

def rfft2(buffer: vd.RFFTBuffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False):
    assert len(buffer.real_shape) == 2 or len(buffer.real_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    rfft(buffer, cmd_stream=cmd_stream, print_shader=print_shader)
    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=len(buffer.real_shape) - 2)

def rfft3(buffer: vd.RFFTBuffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False):
    assert len(buffer.real_shape) == 3, 'Buffer must have 3 dimensions'

    rfft(buffer, cmd_stream=cmd_stream, print_shader=print_shader)
    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=1)
    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=0)

def irfft(buffer: vd.RFFTBuffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False, name: str = None, normalize: bool = True):
    ifft(buffer, buffer_shape=buffer.real_shape, cmd_stream=cmd_stream, print_shader=print_shader, name=name, normalize=normalize, r2c=True)