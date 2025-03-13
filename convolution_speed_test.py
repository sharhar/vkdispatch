import vkdispatch as vd
import vkdispatch.codegen as vc
from matplotlib import pyplot as plt
import numpy as np

import time
import tqdm

side_len = 512

batch_count = 200
batch_num = 20

feature_count = 10
kernel_count = 4

input_buffer = vd.Buffer((feature_count, side_len, side_len), vd.float32)
kernel_buffer = vd.RFFTBuffer((feature_count * kernel_count, side_len, side_len))
output_buffer = vd.RFFTBuffer((feature_count * kernel_count, side_len, side_len))

cmd_stream_convolution = vd.CommandStream()
vd.convolve_2Dreal(output_buffer, kernel_buffer, input=input_buffer, normalize=True, cmd_stream=cmd_stream_convolution)

cmd_stream_fft = vd.CommandStream()

@vd.shader("output.size // 2")
def do_multiply(output : vc.Buffer[vc.c64], kernel : vc.Buffer[vc.c64]):
    tid = vc.global_invocation().x
    output[tid] = vc.mult_c64(output[tid], kernel[tid])

print(output_buffer.shape)

vd.rfft(output_buffer, axes=[1, 2], cmd_stream=cmd_stream_fft)

do_multiply(output_buffer, kernel_buffer, cmd_stream=cmd_stream_fft)

vd.irfft(output_buffer, axes=[1, 2], cmd_stream=cmd_stream_fft, normalize=True)

start_time = time.time()

for _ in tqdm.tqdm(range(batch_count)):
    cmd_stream_fft.submit(batch_num)

output_buffer.read()

print("FFT time:", time.time() - start_time)

start_time = time.time()

for _ in tqdm.tqdm(range(batch_count)):
    cmd_stream_convolution.submit(batch_num)

output_buffer.read()

print("Convolution time:", time.time() - start_time)