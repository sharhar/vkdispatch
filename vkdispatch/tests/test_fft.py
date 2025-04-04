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
                test_data.write(data)

                vd.fft.fft(test_data, axis=axis)

                assert np.allclose(np.fft.fft(data, axis=axis), test_data.read(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_fft_2d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(2)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            test_data = vd.Buffer(data.shape, vd.complex64)

            test_data.write(data)

            vd.fft.fft2(test_data)

            assert np.allclose(np.fft.fft2(data), test_data.read(0), atol=1e-2)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_fft_3d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = 3
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            test_data = vd.Buffer(data.shape, vd.complex64)

            test_data.write(data)

            vd.fft.fft3(test_data)

            assert np.allclose(np.fft.fftn(data), test_data.read(0), atol=5e-2)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_ifft_1d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(1)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            test_data = vd.Buffer(data.shape, vd.complex64)

            for axis in range(dims):
                test_data.write(data)

                vd.fft.ifft(test_data, axis=axis)

                assert np.allclose(np.fft.ifft(data, axis=axis), test_data.read(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_ifft_2d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(2)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            test_data = vd.Buffer(data.shape, vd.complex64)

            test_data.write(data)

            vd.fft.ifft2(test_data)

            assert np.allclose(np.fft.ifft2(data), test_data.read(0), atol=1e-2)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_ifft_3d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = 3
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            test_data = vd.Buffer(data.shape, vd.complex64)

            test_data.write(data)

            vd.fft.ifft3(test_data)

            assert np.allclose(np.fft.ifftn(data), test_data.read(0), atol=5e-2)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_rfft_1d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(1)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.float32)
            test_data = vd.RFFTBuffer(data.shape)

            test_data.write_real(data)

            vd.fft.rfft(test_data)

            assert np.allclose(np.fft.rfft(data), test_data.read_fourier(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_rfft_2d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(2)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.float32)
            test_data = vd.RFFTBuffer(data.shape)

            test_data.write_real(data)

            vd.fft.rfft2(test_data)

            assert np.allclose(np.fft.rfft2(data), test_data.read_fourier(0), atol=1e-2)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_rfft_3d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = 3
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.float32)
            test_data = vd.RFFTBuffer(data.shape)

            test_data.write_real(data)

            vd.fft.rfft3(test_data)

            assert np.allclose(np.fft.rfftn(data), test_data.read_fourier(0), atol=5e-2)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_irfft_1d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(1)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.float32)

            test_data = vd.asrfftbuffer(data)

            vd.fft.rfft(test_data)
            vd.fft.irfft(test_data)

            assert np.allclose(data, test_data.read_real(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_irfft_2d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(2)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.float32)

            test_data = vd.asrfftbuffer(data)

            vd.fft.rfft2(test_data)
            vd.fft.irfft2(test_data)

            assert np.allclose(data, test_data.read_real(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_irfft_3d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = 3
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.float32)

            test_data = vd.asrfftbuffer(data)

            vd.fft.rfft3(test_data)
            vd.fft.irfft3(test_data)

            assert np.allclose(data, test_data.read_real(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])


def test_convolution_2d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(2)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            data2 = np.random.rand(*current_shape).astype(np.complex64)

            test_data = vd.asbuffer(data)
            kernel_data = vd.asbuffer(data2)

            vd.fft.fft2(kernel_data)
            vd.fft.convolve2D(test_data, kernel_data)

            reference_data = numpy_convolution(data, data2)

            assert np.allclose(reference_data, test_data.read(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_convolution_2d_real():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(2)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.float32)
            data2 = np.random.rand(*current_shape).astype(np.float32)

            test_data = vd.asrfftbuffer(data)
            kernel_data = vd.asrfftbuffer(data2)

            vd.fft.rfft2(kernel_data)
            vd.fft.convolve2DR(test_data, kernel_data)

            reference_data = numpy_convolution(data, data2).real

            assert np.allclose(reference_data, test_data.read_real(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13]) 