import vkdispatch as vd
import numpy as np

from matplotlib import pyplot as plt

vd.initialize(debug_mode=True)

N = 64
B = 32
signal = np.zeros((B, N,), dtype=np.complex64)

signal[:, N//4:N//3] = 1

test_data = vd.asbuffer(signal)

vd.fft.convolve(test_data, print_shader=True)

data = test_data.read(0)

diff_arr = np.abs(data - signal)

arg_max = np.argmax(diff_arr)

index_2d = np.unravel_index(arg_max, diff_arr.shape)

print(data[index_2d], signal[index_2d])

plt.imshow(np.abs(signal))
plt.colorbar()
plt.show()

plt.imshow(np.abs(data - signal))
plt.colorbar()
plt.show()

print(np.allclose(data, signal, atol=1e-2))
