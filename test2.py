import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

from matplotlib import pyplot as plt

vd.initialize(debug_mode=True)

def make_random_complex_signal(shape):
    r = np.random.random(size=shape)
    i = np.random.random(size=shape)
    return (r + i * 1j).astype(np.complex64)

def make_square_signal(shape):
    signal = np.zeros(shape)
    signal[shape[0]//4:3*shape[0]//4, shape[1]//4:3*shape[1]//4] = 1
    return signal

def make_gaussian_signal(shape):
    x = np.linspace(-5, 5, shape[0])
    y = np.linspace(-5, 5, shape[1])
    xx, yy = np.meshgrid(x, y)
    signal = np.exp(-xx**2 - yy**2)
    return signal

# Create a 2D buffer
signal_2d = make_gaussian_signal((50, 50)) #make_random_complex_signal((50, 50))

test_img = vd.Buffer(signal_2d.shape, vd.complex64)
test_img.write(signal_2d.astype(np.complex64))

#test_img = vd.asbuffer(signal_2d.astype(np.complex64))

# Perform an FFT on the buffer
vd.fft(test_img)

plt.imshow(np.abs(test_img.read(0))) # - np.abs(np.fft.fft2(signal_2d)))
plt.colorbar()
plt.show()

#plt.imshow(np.abs(np.fft.fft2(signal_2d)))
#plt.colorbar()
#plt.show()

#assert np.allclose(test_img.read(0), np.fft.fft2(signal_2d), atol=0.001)