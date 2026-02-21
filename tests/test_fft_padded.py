import vkdispatch as vd
import numpy as np
import random

from typing import List

TEST_COUNT = 4

def numpy_convolution(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(
        np.fft.fft2(signal).astype(np.complex64)
        *
        np.fft.fft2(kernel).astype(np.complex64).conjugate()
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
    return all([dim <= max_fft_size for dim in fft_dims]) and np.prod(fft_dims) * vd.complex64.item_size < 2 ** 20

def apply_zeros_to_numpy(data: np.ndarray, axis: int, signal_start: int, signal_end: int) -> np.ndarray:
    zeroed_data = data.copy()
    zeroed_data_slices = [slice(None)] * data.ndim
    zeroed_data_slices[axis] = slice(0, signal_start)
    zeroed_data[tuple(zeroed_data_slices)] = 0
    zeroed_data_slices[axis] = slice(signal_end, data.shape[axis])
    zeroed_data[tuple(zeroed_data_slices)] = 0

    return zeroed_data

def test_fft_1d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(TEST_COUNT):
        dims = pick_dim_count(1)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            test_data = vd.Buffer(data.shape, vd.complex64)

            for axis in range(dims):
                test_data.write(data)

                signal_start = np.random.randint(0, data.shape[axis]-1)
                signal_end = np.random.randint(signal_start + 1, data.shape[axis] + 1)

                vd.fft.fft(test_data, axis=axis, input_signal_range=(signal_start, signal_end))
                
                zeroed_data = apply_zeros_to_numpy(data, axis, signal_start, signal_end)

                assert np.allclose(np.fft.fft(zeroed_data, axis=axis), test_data.read(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

    vd.fft.cache_clear()



def test_rfft_1d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(TEST_COUNT):
        dims = pick_dim_count(1)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.float32)
            test_data = vd.RFFTBuffer(data.shape)

            test_data.write_real(data)

            signal_start = np.random.randint(0, data.shape[-1]-1)
            signal_end = np.random.randint(signal_start + 1, data.shape[-1] + 1)

            vd.fft.fft(test_data, buffer_shape=test_data.real_shape, r2c=True, input_signal_range=(signal_start, signal_end))

            zeroed_data = apply_zeros_to_numpy(data, -1, signal_start, signal_end)

            assert np.allclose(np.fft.rfft(zeroed_data), test_data.read_fourier(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

    vd.fft.cache_clear()
