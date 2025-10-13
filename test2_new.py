import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

SIZE = 512

buffer = vd.Buffer((SIZE, SIZE), vd.complex64)
kernel = vd.Buffer((SIZE, SIZE), vd.complex64)

#vd.fft.convolve2D(buffer, kernel) #, print_shader=True)

#exit()

# make a square and circle signal in numpy
x = np.linspace(-1, 1, SIZE)
y = np.linspace(-1, 1, SIZE)
X, Y = np.meshgrid(x, y)
#signal = np.zeros((SIZE, SIZE), dtype=np.complex64)
#signal[np.abs(X) < 0.5] = 1.0 + 0j

#signal2 = np.zeros((SIZE, SIZE), dtype=np.complex64)
#signal2[np.sqrt(X**2 + Y**2) < 0.5] = 1.0 + 0j

signal = np.random.rand(SIZE, SIZE).astype(np.complex64)
signal2 = np.random.rand(SIZE, SIZE).astype(np.complex64)

buffer.write(signal)
kernel.write(signal2)

# perform convolution in numpy for validation
f_signal = np.fft.fft2(signal).astype(np.complex64)
f_kernel = np.fft.fft2(signal2).astype(np.complex64).conjugate()
f_convolved = f_signal * f_kernel
convolved = np.fft.ifft2(f_convolved.astype(np.complex64))

#np.save("signal.npy", signal)
#np.save("kernel.npy", signal2)
#np.save("convolved.npy", convolved)
#np.save("convolved.npy", np.fft.fft(convolved))

vd.fft.fft2(kernel)
vd.fft.fft(buffer)
vd.fft.convolve(buffer, kernel, axis=0, print_shader=True)
vd.fft.ifft(buffer)

vk_convolved = buffer.read(0)

#np.save("vk_convolved.npy", vk_convolved)
#np.save("vk_convolved_fft.npy", np.fft.fft(vk_convolved))

#np.save("diff.npy", (vk_convolved - convolved))
#np.save("diff_fft.npy", (np.fft.fft(vk_convolved) - np.fft.fft(convolved)))

assert np.allclose(vk_convolved, convolved, atol=1e-3)