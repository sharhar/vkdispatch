import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

from typing import Tuple

#vd.initialize(backend="vulkan")

def make_shape(fft_size: int, data_size: int) -> Tuple[int, ...]:
    total_square_size = fft_size * fft_size
    assert data_size % total_square_size == 0, "Data size must be a multiple of fft_size squared"
    return (data_size // total_square_size, fft_size, fft_size)

def make_random_data(fft_size: int, run_index: int, data_size: int, seed: int = 1337) -> np.ndarray:
    shape = make_shape(fft_size, data_size)
    rng = np.random.default_rng(seed + fft_size * 1000 + run_index)

    real = rng.standard_normal(shape).astype(np.float32)
    imag = rng.standard_normal(shape).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)

def compute_metrics(reference: np.ndarray, result: np.ndarray):
    reference64 = reference.astype(np.complex128, copy=False)
    result64 = result.astype(np.complex128, copy=False)

    delta = result64 - reference64
    abs_delta = np.abs(delta)
    abs_reference = np.abs(reference64)

    eps = 1e-12
    relative_l2 = np.linalg.norm(delta.ravel()) / max(np.linalg.norm(reference64.ravel()), eps)
    max_relative = np.max(abs_delta / np.maximum(abs_reference, eps))
    max_absolute = np.max(abs_delta)

    return float(relative_l2), float(max_relative), float(max_absolute)

@vd.map
def kernel_mapping(scale_factor: vc.Var[vc.f32]):
    read_op = vd.fft.read_op()
    read_op.register[:] = read_op.register * scale_factor

fft_size = 4096
data_size = 16 * 1024 * 1024

input_data = make_random_data(fft_size, 0, data_size)
reference = np.fft.fft(input_data)

shape = make_shape(fft_size, data_size)

buffer = vd.buffer_c64(shape) #Buffer(shape, var_type=vd.complex64)

buffer.write(input_data)
#vd.fft.fft(buffer, print_shader=True)
vd.fft.convolve(buffer, np.random.rand(), kernel_map=kernel_mapping, print_shader=True)
result_data = buffer.read(0)

#print(compute_metrics(reference, result_data))