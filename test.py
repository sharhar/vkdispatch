import vkdispatch as vd
import numpy as np
import tqdm
import time

batch_count = 1000
batch_size = 60

vd.initialize(debug_mode=True)

buffer = vd.Buffer((2 ** 10, 2 ** 10), var_type=vd.complex64)
#kernel = vd.RFFTBuffer((2 ** 9, 2 ** 9))
#buffer = vd.RFFTBuffer((650, 169))

cmd_stream_fft = vd.CommandStream()

vd.fft.fft(buffer, axis=0, cmd_stream=cmd_stream_fft, print_shader=True)

#vd.fft.convolve2DR(buffer, kernel, cmd_stream=cmd_stream_fft)

cmd_stream_vkfft = vd.CommandStream()

vd.vkfft.fft(buffer, cmd_stream=cmd_stream_vkfft, axes=[0]) #, keep_shader_code=True)

#vd.vkfft.convolve_2Dreal(buffer, kernel, cmd_stream=cmd_stream_vkfft)

buffer.read()

start_time = time.time()

for _ in tqdm.tqdm(range(batch_count)):
    cmd_stream_fft.submit(batch_size)

buffer.read()

print(f"FFT: {batch_count * batch_size / (time.time() - start_time):.2f} FFT/s")

start_time = time.time()

for _ in tqdm.tqdm(range(batch_count)):
    cmd_stream_vkfft.submit(batch_size)
    #cmd_stream_vkfft.submit(1)
    #exit()


buffer.read()

print(f"VkFFT: {batch_count * batch_size / (time.time() - start_time):.2f} FFT/s")