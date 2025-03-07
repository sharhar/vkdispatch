import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

from matplotlib import pyplot as plt

vd.initialize(debug_mode=True)

side_length = 600

signal_shape = (side_length, side_length)

x = np.linspace(-1, 1, signal_shape[0])
y = np.linspace(-1, 1, signal_shape[1])
x, y = np.meshgrid(x, y)
d = np.sqrt(x*x + y*y)
sigma, mu = 0.02, 0.0
gaussian_signal = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))

mid_idx = side_length // 2

square_signal = np.zeros(signal_shape)
square_signal[mid_idx - 64:mid_idx + 64, mid_idx - 64:mid_idx + 64] = 1

dot_signal = np.zeros(signal_shape, dtype=np.float32)
dot_signal[mid_idx, mid_idx] = 1

input_signal = np.fft.fftshift(dot_signal).astype(np.float32)
kernel_signal = np.fft.fftshift(square_signal).astype(np.float32)

convolved_signal = np.fft.ifftshift(np.fft.ifft2(
    (np.fft.fft2(input_signal).astype(np.complex64) 
    * np.fft.fft2(kernel_signal).astype(np.complex64))
    .astype(np.complex64)))

input_buffer = vd.asrfftbuffer(input_signal)
kernel_buffer = vd.asrfftbuffer(kernel_signal.reshape(1, side_length, -1))
output_buffer = vd.asrfftbuffer(np.ones(signal_shape).astype(np.float32))

vd.prepare_convolution_kernel(kernel_buffer)
vd.convolve_2d(input_buffer, kernel_buffer) #, input_buffer)

result = np.fft.ifftshift(input_buffer.read_real(0))

plt.imshow(result / side_length - np.abs(convolved_signal))
plt.show()

exit()