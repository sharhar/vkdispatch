import vkdispatch as vd

import numpy as np

from matplotlib import pyplot as plt

H, W = 18, 24  # Odd dimensions
x = np.linspace(-5, 5, W)
y = np.linspace(-5, 5, H)
X, Y = np.meshgrid(x, y)
signal = np.exp(-X**2 - Y**2)  # Gaussian function

print(signal.shape, signal.dtype)

# Compute its Fourier transform
F_signal = np.fft.fft2(signal)

Fr_signal = np.fft.rfft2(signal)

padded_signal = np.zeros(shape=(H, W + 2), dtype=np.float32)
padded_signal[:, :W] = signal

signal_buffer = vd.asbuffer(padded_signal)

cmd_stream = vd.CommandStream()

fft_plan = vd.FFTPlan((H, W), do_r2c=True)

fft_plan.record_forward(cmd_stream, signal_buffer)

cmd_stream.submit()

raw_data = signal_buffer.read(0)

raw_data.view(np.complex64)

plt.imshow(np.abs(raw_data.view(np.complex64) - Fr_signal))
plt.show()