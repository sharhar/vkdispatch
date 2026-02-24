import vkdispatch as vd

from .shader_factories import make_fft_shader, make_convolution_shader, make_transpose_shader, get_transposed_size

from typing import Tuple, Union, Optional

def fft_src(
        buffer_shape: Tuple,
        axis: int = None,
        inverse: bool = False,
        normalize_inverse: bool = True,
        r2c: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        input_signal_range: Union[Tuple[Optional[int], Optional[int]], None] = None,
        line_numbers: bool = False) -> vd.ShaderSource:

    fft_shader = make_fft_shader(
        tuple(buffer_shape),
        axis,
        inverse=inverse,
        normalize_inverse=normalize_inverse,
        r2c=r2c,
        input_map=input_map,
        output_map=output_map,
        input_signal_range=input_signal_range)

    return fft_shader.get_src(line_numbers=line_numbers)

def fft2_src(buffer_shape: Tuple, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    assert len(buffer_shape) == 2 or len(buffer_shape) == 3, 'Buffer Shape must have 2 or 3 dimensions'

    return (
        fft_src(axis=len(buffer_shape) - 2, input_map=input_map),
        fft_src(axis=len(buffer_shape) - 1, output_map=output_map)
    )

def fft3_src(buffer_shape: Tuple, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    assert len(buffer_shape) == 3, 'Buffer must have 3 dimensions'

    return (
        fft_src(buffer_shape, axis=0, input_map=input_map),
        fft_src(buffer_shape, axis=1),
        fft_src(buffer_shape, axis=2, output_map=output_map)
    )


def ifft_src(
        buffer_shape: Tuple,
        axis: int = None,
        normalize: bool = True,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None):
    return fft_src(buffer_shape, axis=axis, inverse=True, normalize_inverse=normalize, input_map=input_map, output_map=output_map)

def ifft2_src(buffer_shape: Tuple, normalize: bool = True, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    assert len(buffer_shape) == 2 or len(buffer_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    return (
        ifft_src(buffer_shape, axis=len(buffer_shape) - 2, normalize=normalize, input_map=input_map),
        ifft_src(buffer_shape, axis=len(buffer_shape) - 1, normalize=normalize, output_map=output_map)
    )

def ifft3_src(buffer_shape: Tuple, normalize: bool = True, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    assert len(buffer_shape) == 3, 'Buffer must have 3 dimensions'

    return (
        ifft_src(buffer_shape, axis=0, normalize=normalize, input_map=input_map),
        ifft_src(buffer_shape, axis=1, normalize=normalize),
        ifft_src(buffer_shape, axis=2, normalize=normalize, output_map=output_map)
    )


def rfft_src(buffer_shape: Tuple):
    return fft_src(buffer_shape, r2c=True)

def rfft2_src(buffer_shape: Tuple):
    assert len(buffer_shape) == 2 or len(buffer_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    return (
        rfft_src(buffer_shape),
        fft_src(buffer_shape, axis=len(buffer_shape) - 2)
    )

def rfft3_src(buffer_shape: Tuple):
    assert len(buffer_shape) == 3, 'Buffer must have 3 dimensions'

    return (
        rfft_src(buffer_shape),
        fft_src(buffer_shape, axis=1),
        fft_src(buffer_shape, axis=0)
    )

def irfft_src(buffer_shape: Tuple, normalize: bool = True):
    return fft_src(buffer_shape, inverse=True, normalize_inverse=normalize, r2c=True)

def irfft2_src(buffer_shape: Tuple, normalize: bool = True):
    assert len(buffer_shape) == 2 or len(buffer_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    return (
        ifft_src(buffer_shape, axis=len(buffer_shape) - 2, normalize=normalize),
        irfft_src(buffer_shape, normalize=normalize)
    )

def irfft3_src(buffer_shape: Tuple, normalize: bool = True):
    assert len(buffer_shape) == 3, 'Buffer must have 3 dimensions'

    return (
        ifft_src(buffer_shape, axis=0, normalize=normalize),
        ifft_src(buffer_shape, axis=1, normalize=normalize),
        irfft_src(buffer_shape, normalize=normalize)
    )

def convolve_src(
        buffer_shape: Tuple,
        kernel_map: vd.MappingFunction = None,
        kernel_num: int = 1,
        axis: int = None,
        normalize: bool = True,
        transposed_kernel: bool = False,
        kernel_inner_only: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        input_signal_range: Union[Tuple[Optional[int], Optional[int]], None] = None,
        line_numbers: bool = False) -> vd.ShaderSource:

    fft_shader = make_convolution_shader(
        tuple(buffer_shape),
        kernel_map,
        kernel_num,
        axis,
        transposed_kernel=transposed_kernel,
        kernel_inner_only=kernel_inner_only,
        normalize=normalize,
        input_map=input_map,
        output_map=output_map,
        input_signal_range=input_signal_range)

    return fft_shader.get_src(line_numbers=line_numbers)

def convolve2D_src(
        buffer_shape: Tuple,
        kernel_map: vd.MappingFunction = None,
        normalize: bool = True,
        transposed_kernel: bool = False,
        kernel_inner_only: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None):

    assert len(buffer_shape) == 2 or len(buffer_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    return (
        fft_src(buffer_shape, input_map=input_map),
        convolve_src(
            buffer_shape,
            kernel_map=kernel_map,
            transposed_kernel=transposed_kernel,
            kernel_inner_only=kernel_inner_only,
            axis=len(buffer_shape) - 2,
            normalize=normalize
        ),
        ifft_src(buffer_shape, normalize=normalize, output_map=output_map)
    )

def convolve2DR_src(
        buffer_shape: Tuple,
        kernel_map: vd.MappingFunction = None,
        transposed_kernel: bool = False,
        kernel_inner_only: bool = False,
        normalize: bool = True):
    
    assert len(buffer_shape) == 2 or len(buffer_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    return (
        rfft_src(buffer_shape),
        convolve_src(
            buffer_shape,
            kernel_map=kernel_map,
            transposed_kernel=transposed_kernel,
            kernel_inner_only=kernel_inner_only,
            axis=len(buffer_shape) - 2,
            normalize=normalize
        ),
        irfft_src(buffer_shape, normalize=normalize)
    )

def transpose_src(
        buffer_shape: Tuple,
        axis: int = None,
        kernel_inner_only: bool = False,
        line_numbers: bool = False) -> vd.Buffer:
    
    transpose_shader = make_transpose_shader(
        tuple(buffer_shape),
        axis=axis,
        kernel_inner_only=kernel_inner_only
    )

    return transpose_shader.get_src(line_numbers=line_numbers)


def fft_print_src(
        buffer_shape: Tuple,
        axis: int = None,
        inverse: bool = False,
        normalize_inverse: bool = True,
        r2c: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        input_signal_range: Union[Tuple[Optional[int], Optional[int]], None] = None,
        line_numbers: bool = False) -> vd.ShaderSource:

    print(fft_src(
        buffer_shape,
        axis,
        inverse=inverse,
        normalize_inverse=normalize_inverse,
        r2c=r2c,
        input_map=input_map,
        output_map=output_map,
        input_signal_range=input_signal_range,
        line_numbers=line_numbers))

def fft2_print_src(buffer_shape: Tuple, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    srcs = fft2_src(buffer_shape, input_map=input_map, output_map=output_map)
    print(f"// FFT Stage 1 (axis {len(buffer_shape) - 2}):\n{srcs[0]}\n// FFT Stage 2 (axis {len(buffer_shape) - 1}):\n{srcs[1]}")

def fft3_print_src(buffer_shape: Tuple, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    srcs = fft3_src(buffer_shape, input_map=input_map, output_map=output_map)
    print(f"// FFT Stage 1 (axis 0):\n{srcs[0]}\n// FFT Stage 2 (axis 1):\n{srcs[1]}\n// FFT Stage 3 (axis 2):\n{srcs[2]}")

def ifft_print_src(
        buffer_shape: Tuple,
        axis: int = None,
        normalize: bool = True,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None):
    print(ifft_src(buffer_shape, axis=axis, normalize=normalize, input_map=input_map, output_map=output_map))

def ifft2_print_src(buffer_shape: Tuple, normalize: bool = True, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    srcs = ifft2_src(buffer_shape, normalize=normalize, input_map=input_map, output_map=output_map)
    print(f"// IFFT Stage 1 (axis {len(buffer_shape) - 2}):\n{srcs[0]}\n// IFFT Stage 2 (axis {len(buffer_shape) - 1}):\n{srcs[1]}")

def ifft3_print_src(buffer_shape: Tuple, normalize: bool = True, input_map: vd.MappingFunction = None, output_map: vd.MappingFunction = None):
    srcs = ifft3_src(buffer_shape, normalize=normalize, input_map=input_map, output_map=output_map)
    print(f"// IFFT Stage 1 (axis 0):\n{srcs[0]}\n// IFFT Stage 2 (axis 1):\n{srcs[1]}\n// IFFT Stage 3 (axis 2):\n{srcs[2]}")

def rfft_print_src(buffer_shape: Tuple):
    print(rfft_src(buffer_shape))

def rfft2_print_src(buffer_shape: Tuple):
    srcs = rfft2_src(buffer_shape)
    print(f"// RFFT Stage 1:\n{srcs[0]}\n// RFFT Stage 2 (axis {len(buffer_shape) - 2}):\n{srcs[1]}")

def rfft3_print_src(buffer_shape: Tuple):
    srcs = rfft3_src(buffer_shape)
    print(f"// RFFT Stage 1:\n{srcs[0]}\n// RFFT Stage 2 (axis 1):\n{srcs[1]}\n// RFFT Stage 3 (axis 0):\n{srcs[2]}")

def irfft_print_src(buffer_shape: Tuple, normalize: bool = True):
    print(irfft_src(buffer_shape, normalize=normalize))

def irfft2_print_src(buffer_shape: Tuple, normalize: bool = True):
    srcs = irfft2_src(buffer_shape, normalize=normalize)
    print(f"// IRFFT Stage 1 (axis {len(buffer_shape) - 2}):\n{srcs[0]}\n// IRFFT Stage 2:\n{srcs[1]}")

def irfft3_print_src(buffer_shape: Tuple, normalize: bool = True):
    srcs = irfft3_src(buffer_shape, normalize=normalize)
    print(f"// IRFFT Stage 1 (axis 0):\n{srcs[0]}\n// IRFFT Stage 2 (axis 1):\n{srcs[1]}\n// IRFFT Stage 3:\n{srcs[2]}")

def convolve_print_src(
        buffer_shape: Tuple,
        kernel_map: vd.MappingFunction = None,
        kernel_num: int = 1,
        axis: int = None,
        normalize: bool = True,
        transposed_kernel: bool = False,
        kernel_inner_only: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        input_signal_range: Union[Tuple[Optional[int], Optional[int]], None] = None,
        line_numbers: bool = False) -> vd.ShaderSource:

    print(convolve_src(
        buffer_shape,
        kernel_map=kernel_map,
        kernel_num=kernel_num,
        axis=axis,
        normalize=normalize,
        transposed_kernel=transposed_kernel,
        kernel_inner_only=kernel_inner_only,
        input_map=input_map,
        output_map=output_map,
        input_signal_range=input_signal_range,
        line_numbers=line_numbers
    ))

def convolve2D_print_src(
        buffer_shape: Tuple,
        kernel_map: vd.MappingFunction = None,
        normalize: bool = True,
        transposed_kernel: bool = False,
        kernel_inner_only: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None):
    srcs = convolve2D_src(
        buffer_shape,
        kernel_map=kernel_map,
        normalize=normalize,
        transposed_kernel=transposed_kernel,
        kernel_inner_only=kernel_inner_only,
        input_map=input_map,
        output_map=output_map
    )
    print(f"// FFT Stage (axis {len(buffer_shape) - 2}):\n{srcs[0]}\n// Convolution Stage (axis {len(buffer_shape) - 2}):\n{srcs[1]}\n// IFFT Stage:\n{srcs[2]}")

def convolve2DR_print_src(
        buffer_shape: Tuple,
        kernel_map: vd.MappingFunction = None,
        transposed_kernel: bool = False,
        kernel_inner_only: bool = False,
        normalize: bool = True):
    srcs = convolve2DR_src(
        buffer_shape,
        kernel_map=kernel_map,
        transposed_kernel=transposed_kernel,
        kernel_inner_only=kernel_inner_only,
        normalize=normalize
    )
    print(f"// RFFT Stage:\n{srcs[0]}\n// Convolution Stage (axis {len(buffer_shape) - 2}):\n{srcs[1]}\n// IRFFT Stage:\n{srcs[2]}")

def transpose_print_src(
        buffer_shape: Tuple,
        axis: int = None,
        kernel_inner_only: bool = False,
        line_numbers: bool = False) -> vd.Buffer:
    
    print(transpose_src(
        buffer_shape,
        axis=axis,
        kernel_inner_only=kernel_inner_only,
        line_numbers=line_numbers
    ))