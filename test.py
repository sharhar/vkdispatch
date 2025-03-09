import vkdispatch as vd
from matplotlib import pyplot as plt
import numpy as np

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

def cpu_convolve_2d(signal_2d, kernel_2d):
    return np.fft.irfft2(
        (np.fft.rfft2(signal_2d).astype(np.complex64) 
        * np.fft.rfft2(kernel_2d).astype(np.complex64))
    .astype(np.complex64))

side_len = 32

signal_2d = np.fft.fftshift(np.abs(make_gaussian_signal((side_len, side_len)))).astype(np.float32)
kernel_2d = np.fft.fftshift(np.abs(make_square_signal((side_len, side_len)))).astype(np.float32).reshape((1, side_len, side_len))

test_img = vd.asrfftbuffer(signal_2d)
kernel_img = vd.asrfftbuffer(kernel_2d)

#plt.imshow(kernel_img.read_real(0)[0])
#plt.colorbar()
#plt.savefig("kernel.png")

vd.prepare_convolution_kernel(kernel_img)

#plt.imshow(kernel_img.read_real(0)[0])
#plt.colorbar()
#plt.savefig("kernel_2.png")

fourier_image = kernel_img.read_real(0)[0]
#plt.imshow(fourier_image)
#plt.colorbar()
#plt.show()

# Perform an FFT on the buffer
vd.convolve_2d(test_img, kernel_img)

reference = np.zeros((side_len, side_len))

result = test_img.read_real(0) / side_len
reference = cpu_convolve_2d(signal_2d, kernel_2d[0])

result = np.fft.ifftshift(result)
reference = np.fft.ifftshift(reference)

print(result.mean())
print(reference.mean())

print((result - reference).mean())

plt.imshow(result) # - reference))
plt.colorbar()
plt.savefig("diff.png")
