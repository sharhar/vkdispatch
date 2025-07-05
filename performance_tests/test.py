import vkdispatch as vd
import numpy as np
import tqdm
import time

batch_count = 1000
batch_size = 100

vd.initialize(debug_mode=True)

buffer = vd.Buffer((2 ** 10, 2 ** 12), var_type=vd.complex64)

sync_buffer = vd.Buffer((100, ), vd.float32)

cmd_stream_fft = vd.CommandStream()

axis = 0

vd.fft.fft(buffer, axis=axis, cmd_stream=cmd_stream_fft, print_shader=True)

cmd_stream_vkfft = vd.CommandStream()

print_vkfft_shader = False

vd.vkfft.fft(buffer, cmd_stream=cmd_stream_vkfft, axes=[axis], keep_shader_code=print_vkfft_shader)

sync_buffer.read()

start_time = time.time()

for _ in tqdm.tqdm(range(batch_count)):
    cmd_stream_fft.submit(batch_size)

sync_buffer.read()

print(f"FFT: {batch_count * batch_size / (time.time() - start_time):.2f} FFT/s")

start_time = time.time()

for _ in tqdm.tqdm(range(batch_count)):
    if print_vkfft_shader:
        cmd_stream_vkfft.submit(1)
        sync_buffer.read()
        exit()

    cmd_stream_vkfft.submit(batch_size)


sync_buffer.read()

print(f"VkFFT: {batch_count * batch_size / (time.time() - start_time):.2f} FFT/s")