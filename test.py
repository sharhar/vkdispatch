import vkdispatch as vd
import vkdispatch.codegen as vc
from matplotlib import pyplot as plt
import numpy as np

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

def cpu_convolve_2d(signal_2d, kernel_2d):
    return np.fft.irfft2(
        (np.fft.rfft2(signal_2d).astype(np.complex64) 
        * np.fft.rfft2(kernel_2d).astype(np.complex64))
    .astype(np.complex64))

side_len = 50

signal_2d = (np.abs(make_gaussian_signal((side_len, side_len)))).astype(np.float32)
kernel_2d = (np.abs(make_square_signal((side_len, side_len)))).astype(np.float32).reshape((1, side_len, side_len))

test_img = vd.asrfftbuffer(signal_2d)
kernel_img = vd.asrfftbuffer(kernel_2d)

kernel_img.write(np.ones((1, side_len, side_len + 2), dtype=np.float32))

#vd.prepare_convolution_kernel(kernel_img)

output = vd.RFFTBuffer((side_len, side_len))

# Perform an FFT on the buffer
vd.convolve_2d(test_img, kernel_img, normalize=True)

result = test_img.read(0)

reference = vd.asrfftbuffer(signal_2d)
vd.rfft(reference)
reference = reference.read(0)

#result = np.fft.ifftshift(result)
#reference = np.fft.ifftshift(reference)

print(result.mean())
print(reference.mean())

np.save('result.npy', result)
np.save('reference.npy', reference)

#print((np.abs(result - reference)).mean())

fig, axs = plt.subplots(2, 2)

# Plot the difference between result and reference
axs[0, 0].imshow(result - reference)
axs[0, 0].set_title('Difference')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
 
# Plot the absolute difference between result and reference
axs[0, 1].imshow(np.abs(result - reference))
axs[0, 1].set_title('Absolute Difference')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')

# Plot the reference
axs[1, 0].imshow(reference)
axs[1, 0].set_title('Reference')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')

# Plot the result
axs[1, 1].imshow(result)
axs[1, 1].set_title('Result')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')

plt.show()