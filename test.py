import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

SIZE = 2 ** 6

def numpy_convolution(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(
        np.fft.fft2(signal).astype(np.complex64)
        *
        np.fft.fft2(kernel).astype(np.complex64) # .conjugate()
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

def save_signal(filename: str, data: np.ndarray):
    for ii, layer in enumerate(data):
        np.save(f"data/{filename}_layer{ii}.npy", layer)

current_shape = (2, 11, 5)

#data = np.random.rand(*current_shape).astype(np.complex64)
#data2 = np.random.rand(*current_shape).astype(np.complex64)

data = np.array([make_circle_signal(current_shape[1:], 10 * (i + 1)) for i in range(current_shape[0])]).astype(np.complex64)
data2 = np.array([make_square_signal(current_shape[1:], 50 * (i + 1)) for i in range(current_shape[0])]).astype(np.complex64)

save_signal("input_signal", data)
save_signal("kernel_signal", data2)

test_data = vd.asbuffer(data)
kernel_data = vd.asbuffer(data2)

#    vd.fft.fft2(kernel_data)


#np.save("ffted_kernel.npy", kernel_data.read(0))
#np.save("ffted_kernel_reference.npy", np.fft.fft2(data2).astype(np.complex64))

#kernel_transposed = vd.fft.transpose(kernel_data, axis=1)
#vd.fft.convolve2D(test_data, kernel_transposed, transposed_kernel=True)

#vd.fft.convolve2D(test_data, kernel_transposed, transposed_kernel=True)

vd.vkfft.transpose_kernel2D(kernel_data)
vd.vkfft.convolve2D(test_data, kernel_data, normalize=True)

save_signal("convolved_signal", test_data.read(0))
save_signal("convolved_signal_fourier", np.fft.fft2(test_data.read(0)))

reference_data = numpy_convolution(data, data2)

save_signal("reference_convolved_signal", reference_data)
save_signal("reference_convolved_signal_fourier", np.fft.fft2(reference_data))

save_signal("difference_convolved_signal", reference_data - test_data.read(0))
save_signal("difference_convolved_signal_fourier", np.fft.fft2(reference_data) - np.fft.fft2(test_data.read(0)))

assert np.allclose(reference_data, test_data.read(0), atol=1e-3)