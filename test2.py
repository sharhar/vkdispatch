import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

SIZE = 2 ** 6

def numpy_convolution(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(
        np.fft.fft2(signal).astype(np.complex64)
        *
        np.fft.fft2(kernel).astype(np.complex64).conjugate()
    )


def make_circle_signal(shape, radius):
    center = (shape[0] // 2, shape[1] // 2)
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = dist_from_center <= radius
    array = np.zeros(shape, dtype=np.float32)
    array[mask] = 1.0
    return array

def make_square_signal(shape, size):
    array = np.zeros(shape, dtype=np.float32)
    start_x = (shape[1] - size) // 2
    start_y = (shape[0] - size) // 2
    array[start_y:start_y + size, start_x:start_x + size] = 1.0
    return array

current_shape = (275, 5)

#data = np.random.rand(*current_shape).astype(np.complex64)
#data2 = np.random.rand(*current_shape).astype(np.complex64)

data = make_circle_signal(current_shape, 20).astype(np.complex64)
data2 = make_square_signal(current_shape, 15).astype(np.complex64)

np.save('test_signal.npy', data)
np.save('test_kernel.npy', data2)

test_data = vd.asbuffer(data)
kernel_data = vd.asbuffer(data2)

vd.fft.fft2(kernel_data)

np.save("ffted_kernel.npy", kernel_data.read(0))

np.save("ffted_kernel_reference.npy", np.fft.fft2(data2).astype(np.complex64))

kernel_transposed = vd.fft.transpose(kernel_data, axis=0, print_shader=True)

np.save("transposed_kernel.npy", kernel_transposed.read(0).reshape(275, -1))

print(kernel_data.shape)
print(kernel_transposed.shape)

vd.fft.fft(test_data)
vd.fft.convolve(test_data, kernel_transposed, axis=0, transposed_kernel=True) #, print_shader=True)
vd.fft.ifft(test_data)

np.save("convolved_signal.npy", test_data.read(0))
np.save("convolved_signal_fourier.npy", np.fft.fft2(test_data.read(0)))

reference_data = numpy_convolution(data, data2)

np.save("reference_convolved_signal.npy", reference_data)
np.save("reference_convolved_signal_fourier.npy", np.fft.fft2(reference_data))

assert np.allclose(reference_data, test_data.read(0), atol=1e-3)