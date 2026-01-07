import vkdispatch as vd
import random

from typing import List
import numpy as np

#vd.initialize(log_level=vd.LogLevel.INFO, debug_mode=True)
vd.initialize()

def numpy_convolution(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(
        np.fft.fft2(signal).astype(np.complex64)
        *
        np.fft.fft2(kernel).astype(np.complex64)
    )

def pick_radix_prime():
    return random.choice([2, 3, 5, 7, 11, 13])

def pick_dim_count(min_dim):
    return random.choice(list(range(min_dim, 4)))

def pick_dimention(dims: int):
    if dims == 1:
        return 0

    return random.choice(list(range(dims)))

def check_fft_dims(fft_dims: List[int], max_fft_size: int):
    return all([dim <= max_fft_size for dim in fft_dims]) and np.prod(fft_dims) * vd.complex64.item_size < 2 ** 29

def test_convolution_2d_powers_of_2():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    for i in range(3):
        vd.log_info(f"Starting new 2D convolution test with powers of 2 sizes iter {i+1}/3")

        current_shape = [512, 16, 16]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            data2 = np.random.rand(*current_shape).astype(np.complex64)

            test_data = vd.asbuffer(data)
            kernel_data = vd.asbuffer(data2)

            vd.vkfft.transpose_kernel2D(kernel_data)
            vd.vkfft.convolve2D(test_data, kernel_data, normalize=True)

            reference_data = numpy_convolution(data, data2)

            assert np.allclose(reference_data, test_data.read(0), atol=1e-3)

            current_shape[0] //= 2
            current_shape[1] *= 2
            current_shape[2] *= 2
    
        vd.fft.cache_clear()
    
    vd.log_info("Finished 2D convolution tests with powers of 2 sizes")
