import numpy as np
import matplotlib.pyplot as plt
import vkdispatch as vd
import vkdispatch.codegen as vc

import tqdm

vd.initialize(debug_mode=True, log_level=vd.LogLevel.INFO)

signal_shape = (128, 128)

x = np.linspace(-1, 1, signal_shape[0])
y = np.linspace(-1, 1, signal_shape[1])
x, y = np.meshgrid(x, y)
d = np.sqrt(x*x + y*y)
sigma, mu = 0.1, 0.0
gaussian_signal = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))

square_signal = np.zeros(signal_shape)
square_signal[32:96, 32:96] = 1

convolved_signal = np.fft.ifftshift(np.fft.irfft2(
    (np.fft.rfft2(np.fft.fftshift(gaussian_signal).astype(np.float32)).astype(np.complex64) 
    * np.fft.rfft2(np.fft.fftshift(square_signal).astype(np.float32)).astype(np.complex64))
    .astype(np.complex64)))

gaussian_buffer = vd.asrfftbuffer(np.fft.fftshift(gaussian_signal))

kernel_signal = np.fft.rfft2(np.fft.fftshift(square_signal)).reshape(1, 128, -1)

square_buffer = vd.asbuffer(kernel_signal.astype(np.complex64))

def clasical_convolv():
    vd.rfft(gaussian_buffer)
    vd.rfft(square_buffer)

    @vd.shader("out.size // 2")
    def convolve(out: vc.Buffer[vd.complex64], input: vc.Buffer[vd.complex64]):
        tid = vc.global_invocation().x
        out[tid] *= input[tid]

    convolve(gaussian_buffer, square_buffer)

    vd.irfft(gaussian_buffer)

#clasical_convolv()

vd.convolve_2d(gaussian_buffer, square_buffer)

real_buffer = np.fft.ifftshift(gaussian_buffer.read_real(0)) #/ np.prod(signal_shape)

plt.imshow(real_buffer) #[1:, 1:] - convolved_signal[:-1, :-1])
plt.show()

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