import numpy as np
import matplotlib.pyplot as plt
import vkdispatch as vd
import vkdispatch.codegen as vc

signal = np.ones((16, 16), dtype=np.float32)
#signal[:, 64:128] = 0

signal_buffer = vd.asrfftbuffer(signal)

vd.set_log_level(vd.LogLevel.INFO)

vd.rfft(signal_buffer, padding=[(0, 0), (5, 7)])

fft_singnal = signal_buffer.read_fourier(0)

real_signal = np.fft.irfft2(fft_singnal)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(np.abs(fft_singnal))
axs[0, 0].set_title('Abs FFT')

axs[0, 1].imshow(np.abs(np.fft.ifft(fft_singnal, axis=0)))
axs[0, 1].set_title('Angle FFT')

axs[1, 0].imshow(np.abs(real_signal))
axs[1, 0].set_title('Abs real')

axs[1, 1].imshow(np.angle(real_signal))
axs[1, 1].set_title('angle real')

for ax in axs.flat:
    ax.label_outer()

plt.show()