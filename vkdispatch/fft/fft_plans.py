import vkdispatch as vd
import numpy as np

from .fft_axis import make_fft_stage

def fft(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None, print_shader: bool = False):
    fft_length = buffer.shape[len(buffer.shape) - 1]

    fft_stage = make_fft_stage(
        N=fft_length,
        stride=1
    )

    if print_shader:
        print(fft_stage)

    batch_count = np.round(np.prod(buffer.shape[:len(buffer.shape) - 1])).astype(np.int32)
    workgroup_count = 1

    fft_stage(buffer, cmd_stream=cmd_stream, exec_size=(workgroup_count, batch_count, 1))
    #fft_stage(buffer, cmd_stream=cmd_stream, exec_size=(max(1, fft_length // 8), batch_count, 1))