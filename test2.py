import vkdispatch as vd
import numpy as np

from matplotlib import pyplot as plt

vd.initialize(debug_mode=True)

N = 64
B = 32
signal = np.zeros((B, N,), dtype=np.float32)

signal[:, N//4:N//3] = 1

test_data = vd.asrfftbuffer(signal)

vd.fft.rfft(test_data)
vd.fft.irfft(test_data)

data = test_data.read_real(0)

diff_arr = np.abs(data - signal)

arg_max = np.argmax(diff_arr)

index_2d = np.unravel_index(arg_max, diff_arr.shape)

print(data[index_2d], signal[index_2d])

plt.imshow(np.abs(signal))
plt.colorbar()
plt.show()

plt.imshow(np.abs(data))
plt.colorbar()
plt.show()

print(np.allclose(data, signal, atol=1e-2))


exit()

N = 64

# make square signal
signal = np.zeros((N, N), dtype=np.complex64)
signal[N//4:3*N//4, N//4:3*N//4] = 1

# add random dots to signal
signal += (np.random.rand(N, N) + 1j * np.random.rand(N, N)) / 10

plt.imshow(np.abs(signal))
plt.show()

signal_fft = np.fft.fft(signal.astype(np.complex64), axis=1)

#plt.imshow(np.abs(signal_fft))
#plt.show()

signal_gpu = vd.asbuffer(signal.astype(np.complex64))
vd.fft.fft(signal_gpu)

data = np.fft.ifft(signal_gpu.read(0), axis=1)

#plt.imshow(np.abs(data))
#plt.show()


print(np.allclose(signal_fft, signal_gpu.read(0), atol=1e-2))
