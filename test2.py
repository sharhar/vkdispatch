import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

SIZE = 512

buffer = vd.Buffer((SIZE, SIZE), vd.complex64)
kernel = vd.Buffer((SIZE, SIZE), vd.complex64)

# make a square and circle signal in numpy
x = np.linspace(-1, 1, SIZE)
y = np.linspace(-1, 1, SIZE)
X, Y = np.meshgrid(x, y)
signal = np.zeros((SIZE, SIZE), dtype=np.complex64)
signal[np.abs(X) < 0.5] = 1.0 + 0j

signal2 = np.zeros((SIZE, SIZE), dtype=np.complex64)
signal2[np.sqrt(X**2 + Y**2) < 0.5] = 1.0 + 0j

buffer.write(signal)
kernel.write(signal2)

# perform convolution in numpy for validation
f_signal = np.fft.fft2(signal)
f_kernel = np.fft.fft2(signal2)
f_convolved = f_signal * f_kernel
convolved = np.fft.ifft2(f_convolved)

np.save("signal.npy", signal)
np.save("kernel.npy", signal2)
np.save("convolved.npy", convolved)


vd.fft.convolve2D(buffer, kernel)

vk_convolved = buffer.read(0)

np.save("vk_convolved.npy", vk_convolved)