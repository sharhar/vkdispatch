import vkdispatch as vd
import numpy as np

from matplotlib import pyplot as plt

vd.initialize(debug_mode=True)

N = 30
B = 20

#N = 512
#B = 512
signal = np.zeros((B, N,), dtype=np.float32)

signal[:, N//4:N//3] = 1

#signal = (np.random.rand(B, N) + 1j * np.random.rand(B, N)).astype(np.complex64)

signal_gpu = vd.asrfftbuffer(signal)

vd.fft.rfft(signal_gpu, print_shader=True)
vd.fft.fft(signal_gpu, axis=0)

data = np.array(signal_gpu.read_fourier(0))

reference_data = np.fft.rfft2(signal)

diff_arr = np.abs(data - reference_data)

arg_max = np.argmax(diff_arr)

index_2d = np.unravel_index(arg_max, diff_arr.shape)

print(data[index_2d], reference_data[index_2d])

plt.imshow(np.abs(reference_data))
plt.colorbar()
plt.show()

plt.imshow(np.abs(data - reference_data))
plt.colorbar()
plt.show()

print(np.allclose(data, reference_data, atol=1e-2))


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
