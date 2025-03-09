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



side_len = 64

dot_signal = np.zeros((side_len, side_len))
dot_signal[side_len//2, side_len//2] = 1

gaussian_signal = make_gaussian_signal((side_len, side_len))
square_signal = make_square_signal((side_len, side_len))

signal_2d = (np.abs(square_signal)).astype(np.float32)
kernel_2d = (np.abs(gaussian_signal)).astype(np.float32).reshape((1, side_len, side_len))

#lt.imshow(signal_2d)
#lt.show()

#plt.imshow(kernel_2d[0])
#plt.show()

padded_kernel = np.ones(shape=(1, 2*side_len, side_len)) * -800
padded_kernel[0, :side_len, :] = kernel_2d[0]

kernel_array = np.fft.rfft2(kernel_2d)

test_img = vd.asrfftbuffer(signal_2d)
kernel_img = vd.asrfftbuffer(kernel_2d)

#plt.imshow(np.abs(kernel_img.read_real(0)[0]))
#plt.colorbar()
#plt.savefig("kernel.png")

vd.prepare_convolution_kernel(kernel_img) #, shape=(1, side_len, side_len + 2))

#plt.imshow(kernel_img.read_real(0)[0])
#plt.colorbar()
#plt.show()

@vd.shader("out.size // 2")
def do_mult(out: vc.Buffer[vc.c64], kernel: vc.Buffer[vc.c64]):
    tid = vc.global_invocation().x

    out[tid] = vc.mult_c64(out[tid], kernel[tid])

#vd.rfft(test_img)

#do_mult(test_img, kernel_img)

#vd.irfft(test_img, normalize=True)

# Perform an FFT on the buffer

#vd.rfft(test_img)

reference = np.abs(np.ascontiguousarray(np.fft.rfft2(signal_2d), dtype=np.complex64))

#reference = vd.asrfftbuffer(signal_2d)
#vd.rfft(reference) #, axes=[1])

#vd.convolve_2d(test_img, kernel_img, normalize=True)

vd.rfft(test_img)

result = np.abs(test_img.read_fourier(0))
#reference = cpu_convolve_2d(signal_2d, kernel_2d[0])/ side_len



#reference = np.abs(reference.read_fourier(0))

result = np.fft.ifftshift(result)
reference = np.fft.ifftshift(reference)

print(result.mean())
print(reference.mean())

#print((np.abs(result - reference)).mean())

result_FFT = np.fft.rfft2(result)
reference_FFT = np.fft.rfft2(reference)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image
im1 = ax1.imshow(np.abs(result), cmap='viridis')
ax1.set_title('Image 1')
fig.colorbar(im1, ax=ax1)

# Plot the second image
im2 = ax2.imshow(np.abs(reference), cmap='viridis')
ax2.set_title('Image 2')
fig.colorbar(im2, ax=ax2)

plt.show()

#plt.imshow(np.abs(result_FFT - reference_FFT))
#plt.colorbar()
#plt.show()
