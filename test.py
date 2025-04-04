import vkdispatch as vd
import numpy as np
import tqdm
import time

batch_count = 1000
batch_size = 100

vd.initialize(debug_mode=True)

buffer = vd.RFFTBuffer((2 ** 9, 2 ** 9))
kernel = vd.RFFTBuffer((2 ** 9, 2 ** 9))
#buffer = vd.RFFTBuffer((650, 169))

cmd_stream_fft = vd.CommandStream()

#vd.fft.rfft2(buffer, cmd_stream=cmd_stream_fft)

vd.fft.convolve2DR(buffer, kernel, cmd_stream=cmd_stream_fft)

cmd_stream_vkfft = vd.CommandStream()

#vd.vkfft.rfft(buffer, cmd_stream=cmd_stream_vkfft)

vd.vkfft.convolve_2Dreal(buffer, kernel, cmd_stream=cmd_stream_vkfft)

buffer.read()

start_time = time.time()

for _ in tqdm.tqdm(range(batch_count)):
    cmd_stream_fft.submit(batch_size)

buffer.read()

print(f"FFT: {batch_count * batch_size / (time.time() - start_time):.2f} FFT/s")

start_time = time.time()

for _ in tqdm.tqdm(range(batch_count)):
    cmd_stream_vkfft.submit(batch_size)

buffer.read()

print(f"VkFFT: {batch_count * batch_size / (time.time() - start_time):.2f} FFT/s")