import vkdispatch as vd

import numpy as np

from matplotlib import pyplot as plt

H, W = 18, 24  # Odd dimensions
x = np.linspace(-5, 5, W)
y = np.linspace(-5, 5, H)
X, Y = np.meshgrid(x, y)
signal = np.exp(-X**2 - Y**2)  # Gaussian function

print(signal.shape, signal.dtype)

Fr_signal = np.fft.fft(signal)

signal_buffer = vd.asbuffer(signal.astype(np.complex64))

vd.fft(signal_buffer, axes=[1])

plt.imshow(np.abs(Fr_signal - signal_buffer.read(0)))
plt.show()

#vd.irfft(signal_buffer)

#plt.imshow(signal_buffer.read_real(0) / np.prod(signal_buffer.real_shape) - signal)
#plt.show()


