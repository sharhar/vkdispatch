import numpy as np
import matplotlib.pyplot as plt
import vkdispatch as vd
import vkdispatch.codegen as vc

import tqdm

import time

vd.initialize(debug_mode=True)

buff_size = (512, 512)
kernel_size = (1, buff_size[0], buff_size[1] // 2 + 1)

buffer = vd.RFFTBuffer(buff_size)
kernel = vd.Buffer(kernel_size, vd.complex64)

cmd_stream = None # vd.CommandStream()

def clasical_convolv():
    vd.rfft(buffer, cmd_stream=cmd_stream)

    @vd.shader("out.size // 2")
    def convolve(out: vc.Buffer[vd.complex64], input: vc.Buffer[vd.complex64]):
        tid = vc.global_invocation().x
        out[tid] *= input[tid]

    convolve(buffer, kernel, cmd_stream=cmd_stream)

    vd.irfft(buffer, cmd_stream=cmd_stream)

clasical_convolv()

plt.imshow(np.abs(buffer.read()))
plt.show()

exit()

start_time = time.time()

for _ in tqdm.tqdm(range(2500)):
    cmd_stream.submit(100)

buffer.read(0)

classical_time = time.time() - start_time

cmd_stream.reset()

start_time = time.time()

vd.convolve_2d(buffer, kernel, cmd_stream=cmd_stream)

for _ in tqdm.tqdm(range(2500)):
    cmd_stream.submit(100)

buffer.read(0)

new_time = time.time() - start_time

print(f"Classical time: {classical_time}")
print(f"New time: {new_time}")

buffer.read()

exit()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(np.abs(fft_singnal))
axs[0, 0].set_title('Abs FFT')

axs[0, 1].imshow(np.abs(np.fft.fft(zeroed_signal, axis=0)))
axs[0, 1].set_title('Angle FFT')

axs[1, 0].imshow(np.abs(real_signal))
axs[1, 0].set_title('Abs real')

axs[1, 1].imshow(np.abs(np.fft.ifft(real_signal, axis=1)))
axs[1, 1].set_title('angle real')

for ax in axs.flat:
    ax.label_outer()

plt.show()