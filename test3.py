import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

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

def test_shape(shape):
    data = np.random.rand(*shape).astype(np.complex64)
    test_data = vd.Buffer(data.shape, vd.complex64)

    print(f"Testing FFT with shape {data.shape}")

    for axis in range(len(shape)):

        print(f"Testing axis {axis}")

        test_data.write(data)

        print("Running FFT")

        vd.fft.fft(test_data, axis=axis) #, print_shader=True)

        print("Reading data")

        assert np.allclose(np.fft.fft(data, axis=axis), test_data.read(0), atol=1e-3)


def test_fft_1d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for _ in range(20):
        dims = pick_dim_count(1)
        current_shape = [pick_radix_prime() for _ in range(dims)]

        while check_fft_dims(current_shape, max_fft_size):
            test_shape(current_shape)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

def test_convolution_2d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0])

    for i in range(200):
        dims = 2 # pick_dim_count(2)
        current_shape = [pick_radix_prime() * 2 for _ in range(dims)]

        #current_shape = [308, 338] # (32, 32, 10) #13, 2, 11)

        prev_reference_data = None
        prev_test_data_cpu = None
        prev_kernel_data = None

        while check_fft_dims(current_shape, max_fft_size):
            data = np.random.rand(*current_shape).astype(np.complex64)
            data2 = np.random.rand(*current_shape).astype(np.complex64)

            test_data = vd.asbuffer(data)
            kernel_data = vd.asbuffer(data2)

            print(f"{i} Testing FFT with shape {data.shape} {current_shape}")

            vd.fft.fft2(kernel_data)
            vd.fft.convolve2D(test_data, kernel_data) #, print_shader=True)

            reference_data = numpy_convolution(data, data2)

            test_data_cpu = test_data.read(0)

            print(f"Mean squared error: {np.log(np.sum(np.abs(reference_data - test_data_cpu) ** 2))}")
            print(f"Max error: {np.log(np.max(np.abs(reference_data - test_data_cpu)))}")

            passed_test = np.allclose(reference_data, test_data.read(0), atol=1e-3)

            if not passed_test:
                np.save(f"test_data.npy", test_data_cpu)
                np.save(f"kernel_data.npy", kernel_data.read(0))
                np.save(f"reference_data.npy", reference_data)

                np.save(f"prev_test_data_cpu.npy", prev_test_data_cpu)
                np.save(f"prev_kernel_data.npy", prev_kernel_data)
                np.save(f"prev_reference_data.npy", prev_reference_data)

                test_data_cpu_fft = np.fft.fft(test_data_cpu)
                reference_data_fft = np.fft.fft(reference_data)

                np.save(f"test_data_fft.npy", test_data_cpu_fft)
                np.save(f"reference_data_fft.npy", reference_data_fft)

                test_data_cpu_fft2 = np.fft.fft(test_data_cpu_fft, axis=0)
                reference_data_fft2 = np.fft.fft(reference_data_fft, axis=0)

                test_data_cpu_fft2[0, 0] = 0
                reference_data_fft2[0, 0] = 0

                np.save(f"test_data_fft2.npy", test_data_cpu_fft2)
                np.save(f"reference_data_fft2.npy", reference_data_fft2)
            else:
                prev_test_data_cpu = test_data_cpu
                prev_reference_data = reference_data
                prev_kernel_data = kernel_data.read(0)

            assert passed_test
            
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

            vd.fft.rfft(test_data, print_shader=True)
            vd.fft.irfft(test_data, print_shader=True)

            print(f"Testing FFT with shape {data.shape}")

            assert np.allclose(data, test_data.read_real(0), atol=1e-3)

            current_shape[pick_dimention(dims)] *= random.choice([2, 3, 5, 7, 11, 13])

@vd.shader(exec_size=lambda args: args.atom_coords.shape[0])
def place_atoms(image: Buff[i32], atom_coords: Buff[f32], rot_matrix: Var[m4], pixel_size: Const[f32]):
    ind = vc.global_invocation().x.copy()

    pos = vc.new_vec4() #shader.new(vd.vec4)
    pos.x = -atom_coords[3*ind + 1] / pixel_size
    pos.y = atom_coords[3*ind + 0] / pixel_size
    pos.z = atom_coords[3*ind + 2] / pixel_size
    pos.w = 1

    pos[:] = rot_matrix * pos

    image_ind = vc.new_ivec2()
    image_ind.y = vc.ceil(pos.y).cast_to(vd.int32) + (image.shape.x / 2)
    image_ind.x = vc.ceil(-pos.x).cast_to(vd.int32) + (image.shape.y / 2)

    vc.if_any(image_ind.x < 0, image_ind.x >= image.shape.x, image_ind.y < 0, image_ind.y >= image.shape.y)
    vc.return_statement()
    vc.end()

    vc.atomic_add(image[2 * image_ind.x, 2 * image_ind.y], 1)

#print(place_atoms)

#test_shape((44, 234))

#test_irfft_1d()
#test_fft_1d()

test_convolution_2d()