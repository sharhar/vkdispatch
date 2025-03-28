import vkdispatch as vd
import numpy as np

from .fft_axis import make_fft_stage

def fft(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False, axis: int = None, name: str = None, inverse: bool = False, normalize_inverse: bool = True):
    if axis is None:
        axis = len(buffer.shape) - 1

    total_buffer_length = np.round(np.prod(buffer.shape)).astype(np.int32)

    fft_length = buffer.shape[axis]

    stride = np.round(np.prod(buffer.shape[axis + 1:])).astype(np.int32)
    batch_y_stride = stride * fft_length
    batch_y_count = total_buffer_length // batch_y_stride

    batch_z_stride = 1
    batch_z_count = stride

    fft_stage = make_fft_stage(
        N=fft_length,
        stride=stride,
        batch_y_stride=batch_y_stride,
        batch_z_stride=batch_z_stride,
        name=name
    )

    fft_shader, exec_size = fft_stage.shader(batch_y_count, batch_z_count, inverse=inverse, normalize_inverse=normalize_inverse)

    if print_shader:
        print(fft_shader)

    fft_shader(buffer, cmd_stream=cmd_stream, exec_size=exec_size)

def ifft(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False, axis: int = None, name: str = None, normalize: bool = True):
    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=axis, name=name, inverse=True, normalize_inverse=normalize)

def fft2(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False):
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=len(buffer.shape) - 2)
    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=len(buffer.shape) - 1)

def ifft2(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False, normalize: bool = True):
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    ifft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=len(buffer.shape) - 2, normalize=normalize)
    ifft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=len(buffer.shape) - 1, normalize=normalize)

def fft3(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False):
    assert len(buffer.shape) == 3, 'Buffer must have 3 dimensions'

    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=0)
    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=1)
    fft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=2)

def ifft3(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False, normalize: bool = True):
    assert len(buffer.shape) == 3, 'Buffer must have 3 dimensions'

    ifft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=0, normalize=normalize)
    ifft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=1, normalize=normalize)
    ifft(buffer, cmd_stream=cmd_stream, print_shader=print_shader, axis=2, normalize=normalize)