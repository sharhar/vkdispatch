import vkdispatch as vd
import numpy as np

from matplotlib import pyplot as plt

vd.initialize(debug_mode=True)

"""
array([15.        +0.j        , -0.92387953+0.38268343j,
       -0.70710678+0.70710678j, -0.38268343+0.92387953j,
        0.        +1.j        ,  0.38268343+0.92387953j,
        0.70710678+0.70710678j,  0.92387953+0.38268343j,
        1.        +0.j        ,  0.92387953-0.38268343j,
        0.70710678-0.70710678j,  0.38268343-0.92387953j,
        0.        -1.j        , -0.38268343-0.92387953j,
       -0.70710678-0.70710678j, -0.92387953-0.38268343j])

"""


N = 64

signal = np.ones((N,), dtype=np.complex64)
signal[N//4] = 0

#signal = (np.random.rand(N) + 1j * np.random.rand(N)).astype(np.complex64)

signal_gpu = vd.asbuffer(signal)

vd.fft.fft(signal_gpu) #, print_shader=True)

data = signal_gpu.read(0).reshape(-1, 8)
reference_data = np.fft.fft(signal).reshape(-1, 8)

data = np.round(data, 3)
reference_data = np.round(reference_data, 3)

print(data)

#print(data.flatten()[::2])
#print(data.flatten()[1::2])

print(reference_data)

#print(data - reference_data)
print(np.allclose(data, reference_data, atol=1e-3))


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

plt.imshow(np.abs(signal_fft))
plt.show()

signal_gpu = vd.asbuffer(signal.astype(np.complex64))
vd.fft.fft(signal_gpu)

data = np.fft.ifft(signal_gpu.read(0), axis=1)

plt.imshow(np.abs(data))
plt.show()

print(np.allclose(signal_fft, signal_gpu.read(0), atol=1e-3))
