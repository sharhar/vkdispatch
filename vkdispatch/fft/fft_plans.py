import vkdispatch as vd
import numpy as np

from .fft_axis import make_fft_stage

def fft(buffer: vd.Buffer, cmd_stream: vd.CommandStream = None):
    fft_length = buffer.shape[len(buffer.shape) - 1]

    fft_stage = make_fft_stage(
        N=fft_length,
        stride=1,
        batch_input_stride=fft_length,
        batch_output_stride=fft_length
    )

    batch_count = np.round(np.prod(buffer.shape[:len(buffer.shape) - 1])).astype(np.int32)

    fft_stage(buffer, cmd_stream=cmd_stream, exec_size=(fft_length // 2, batch_count, 1))