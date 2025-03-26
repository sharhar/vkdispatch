import vkdispatch as vd
import numpy as np
import tqdm
import time

batch_count = 1000
batch_size = 2

N = 4096

buffer = vd.Buffer((2 ** 12, 2 ** 12), vd.complex64)

cmd_stream_fft = vd.CommandStream()

#vd.fft.fft(buffer, axis=1, cmd_stream=cmd_stream_fft)
vd.fft.fft2(buffer, cmd_stream=cmd_stream_fft)

cmd_stream_vkfft = vd.CommandStream()

#vd.vkfft.fft(buffer, axes=[1], cmd_stream=cmd_stream_vkfft)
vd.vkfft.fft(buffer, cmd_stream=cmd_stream_vkfft)

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