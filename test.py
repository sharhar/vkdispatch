import vkdispatch as vd
import numpy as np
import random

from typing import List

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

def test_fft_1d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(1)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            test_data = vd.Buffer(data.shape, vd.complex64)

            for axis in range(dims):
                print(current_shape, axis)

                test_data.write(data)

                vd.fft.fft(test_data, axis=axis)

                assert np.allclose(np.fft.fft(data, axis=axis), test_data.read(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

    vd.fft.cache_clear()


def test_rfft_1d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(1)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            print(current_shape)

            data = np.random.rand(*current_shape).astype(np.float32)
            test_data = vd.RFFTBuffer(data.shape)

            test_data.write_real(data)

            vd.fft.rfft(test_data)

            assert np.allclose(np.fft.rfft(data), test_data.read_fourier(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

    vd.fft.cache_clear()


test_fft_1d()

data = np.random.rand(495).astype(np.complex64)
test_data = vd.RFFTBuffer(data.shape)
#print(current_shape, axis)

#test_data.write(data)

vd.fft.rfft(test_data) #, print_shader=True)

exit()

fft_data = test_data.read(0)
np_data = np.fft.fft(data, axis=0)

#print(np_data[0])

np.save("fft_np.npy", np_data.reshape(45, 11))
np.save("fft_vk.npy", fft_data.reshape(45, 11))

assert np.allclose(np_data, fft_data, atol=1e-3)