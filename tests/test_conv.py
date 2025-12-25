import vkdispatch as vd
import numpy as np
import random

from typing import List

TEST_COUNT = 4

def numpy_convolution_1d(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.fft.ifft(
        np.fft.fft(signal).astype(np.complex64)
        *
        np.fft.fft(kernel).astype(np.complex64).conjugate()
    )

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

def test_convolution_1d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(TEST_COUNT):
        dims = pick_dim_count(2)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            data2 = np.random.rand(*current_shape).astype(np.complex64)

            test_data = vd.asbuffer(data)
            kernel_data = vd.asbuffer(data2)

            vd.fft.fft(kernel_data)
            vd.fft.convolve(test_data, kernel_data)

            reference_data = numpy_convolution_1d(data, data2)

            assert np.allclose(reference_data, test_data.read(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])
    
    vd.fft.cache_clear()

def test_convolution_2d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(TEST_COUNT):
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
    
    vd.fft.cache_clear()

def test_convolution_2d_transpose():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    kernel_transposed_buffer = vd.Buffer((2048,), var_type=vd.complex64)

    for _ in range(TEST_COUNT):
        dims = pick_dim_count(2)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            data2 = np.random.rand(*current_shape).astype(np.complex64)

            test_data = vd.asbuffer(data)
            kernel_data = vd.asbuffer(data2)

            transpose_size  = vd.fft.get_transposed_size(
                tuple(current_shape),
                axis=len(kernel_data.shape)-2
            )

            # Allocate new transposed buffer if needed
            if transpose_size > kernel_transposed_buffer.size:
                kernel_transposed_buffer.destroy()
                kernel_transposed_buffer = vd.Buffer((transpose_size,), var_type=vd.complex64)

            vd.fft.fft2(kernel_data)
            vd.fft.transpose(kernel_data, out_buffer=kernel_transposed_buffer, axis=len(kernel_data.shape)-2)
            vd.fft.convolve2D(test_data, kernel_transposed_buffer, transposed_kernel=True)

            reference_data = numpy_convolution(data, data2)

            assert np.allclose(reference_data, test_data.read(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])
    
    vd.fft.cache_clear()

def test_convolution_2d_real():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(TEST_COUNT):
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

    vd.fft.cache_clear()

def test_convolution_2d_inner():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(TEST_COUNT):
        dims = 3
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            data2 = np.random.rand(*current_shape[1:]).astype(np.complex64)

            test_data = vd.asbuffer(data)
            kernel_data = vd.asbuffer(data2)

            vd.fft.fft2(kernel_data)
            vd.fft.convolve2D(
                test_data,
                kernel_data,
                kernel_inner_only=True
            )

            reference_data = numpy_convolution(data, data2)

            assert np.allclose(reference_data, test_data.read(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])
    
    vd.fft.cache_clear()

def test_convolution_2d_transpose_inner():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    kernel_transposed_buffer = vd.Buffer((2048,), var_type=vd.complex64)

    for _ in range(TEST_COUNT):
        dims = 3
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            data2 = np.random.rand(*current_shape[1:]).astype(np.complex64)

            test_data = vd.asbuffer(data)
            kernel_data = vd.asbuffer(data2)

            transpose_size  = vd.fft.get_transposed_size(
                tuple(current_shape),
                axis=len(kernel_data.shape)-2
            )

            # Allocate new transposed buffer if needed
            if transpose_size > kernel_transposed_buffer.size:
                kernel_transposed_buffer.destroy()
                kernel_transposed_buffer = vd.Buffer((transpose_size,), var_type=vd.complex64)

            vd.fft.fft2(kernel_data)
            vd.fft.transpose(
                kernel_data,
                conv_shape=current_shape,
                out_buffer=kernel_transposed_buffer,
                axis=len(kernel_data.shape)-2,
                kernel_inner_only=True
            )
            vd.fft.convolve2D(
                test_data,
                kernel_transposed_buffer,
                transposed_kernel=True,
                kernel_inner_only=True
            )

            reference_data = numpy_convolution(data, data2)

            assert np.allclose(reference_data, test_data.read(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])
    
    vd.fft.cache_clear()

test_convolution_2d_transpose_inner()
