import numpy as np
from matplotlib import pyplot as plt

# make rotated (30 degrees) square signal
signal_2d = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        x = i - 50
        y = j - 50
        x_new = x * np.cos(np.pi/6) - y * np.sin(np.pi/6)
        y_new = x * np.sin(np.pi/6) + y * np.cos(np.pi/6)
        if -10 < x_new < 10 and -10 < y_new < 10:
            signal_2d[i, j] = 1

plt.imshow(signal_2d)
plt.colorbar()
plt.show()

signal_fft = np.fft.fft2(signal_2d)
signal_rfft = np.fft.rfft2(signal_2d)

plt.imshow(np.abs(signal_fft))
plt.colorbar()
plt.show()

plt.imshow(np.abs(signal_rfft))
plt.colorbar()
plt.show()

signal_fft_half0 = signal_fft[:, :51]

plt.imshow(np.abs(signal_fft_half0 - signal_rfft))
plt.colorbar()
plt.show()

