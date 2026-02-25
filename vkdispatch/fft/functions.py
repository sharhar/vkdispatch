import vkdispatch as vd

from .shader_factories import make_fft_shader, make_convolution_shader, make_transpose_shader, get_transposed_size
from .precision import (
    ensure_supported_complex_precision,
    resolve_compute_precision,
    validate_complex_precision,
)

from typing import List, Tuple, Union, Optional


def _validate_map_argument_annotations(map_fn: vd.MappingFunction, map_name: str) -> None:
    for buffer_type in map_fn.buffer_types:
        if not hasattr(buffer_type, "__args__") or len(buffer_type.__args__) != 1:
            raise ValueError(
                f"{map_name} contains an annotation without exactly one type argument: {buffer_type}"
            )


def _resolve_output_precision(
    buffers: Tuple[vd.Buffer, ...],
    output_map: Optional[vd.MappingFunction],
    output_type: Optional[vd.dtype],
) -> Optional[vd.dtype]:
    if output_map is not None:
        if output_type is not None:
            raise ValueError("output_type cannot be provided when output_map is used")
        return None

    resolved_output = buffers[0].var_type if output_type is None else output_type
    validate_complex_precision(resolved_output, arg_name="output_type")
    ensure_supported_complex_precision(resolved_output, role="Output")
    return resolved_output


def _resolve_input_precision(
    input_map: Optional[vd.MappingFunction],
    output_map: Optional[vd.MappingFunction],
    input_type: Optional[vd.dtype],
    output_precision: Optional[vd.dtype],
) -> Optional[vd.dtype]:
    if input_map is not None:
        if input_type is not None:
            raise ValueError("input_type cannot be provided when input_map is used")
        return None

    if output_map is not None:
        if input_type is not None:
            raise ValueError("input_type cannot be provided when output_map is used without input_map")
        return None

    if output_precision is None:
        raise ValueError("output_precision must be provided when output_map is not used")

    resolved_input = output_precision if input_type is None else input_type
    validate_complex_precision(resolved_input, arg_name="input_type")
    ensure_supported_complex_precision(resolved_input, role="Input")

    if resolved_input != output_precision:
        raise ValueError(
            "input_type must match output_type when input_map is None (default FFT path is in-place)"
        )

    return resolved_input


def _resolve_kernel_precision(
    buffers: Tuple[vd.Buffer, ...],
    kernel_map: Optional[vd.MappingFunction],
    kernel_type: Optional[vd.dtype],
) -> Optional[vd.dtype]:
    if kernel_map is not None:
        if kernel_type is not None:
            raise ValueError("kernel_type cannot be provided when kernel_map is used")
        return None

    if len(buffers) < 2:
        raise ValueError("Kernel precision inference requires a kernel buffer argument")

    resolved_kernel = buffers[1].var_type if kernel_type is None else kernel_type
    validate_complex_precision(resolved_kernel, arg_name="kernel_type")
    ensure_supported_complex_precision(resolved_kernel, role="Kernel")
    return resolved_kernel

def fft(
        *buffers: vd.Buffer,
        buffer_shape: Tuple = None,
        graph: vd.CommandGraph = None,
        print_shader: bool = False,
        axis: int = None,
        name: str = None,
        inverse: bool = False,
        normalize_inverse: bool = True,
        r2c: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        output_type: vd.dtype = None,
        input_type: vd.dtype = None,
        compute_type: vd.dtype = None,
        input_signal_range: Union[Tuple[Optional[int], Optional[int]], None] = None):
    
    assert len(buffers) >= 1, "At least one buffer must be provided"

    if input_map is None and output_map is None and len(buffers) != 1:
        raise ValueError("fft() expects exactly one buffer unless input_map/output_map are used")
    
    if buffer_shape is None:
        buffer_shape = buffers[0].shape

    resolved_output_type = _resolve_output_precision(buffers, output_map, output_type)
    resolved_input_type = _resolve_input_precision(input_map, output_map, input_type, resolved_output_type)

    io_precisions: List[vd.dtype] = []
    if output_map is None:
        io_precisions.append(resolved_output_type)
    else:
        _validate_map_argument_annotations(output_map, "output_map")

    if input_map is None:
        if resolved_input_type is not None:
            io_precisions.append(resolved_input_type)
    else:
        _validate_map_argument_annotations(input_map, "input_map")

    resolved_compute_type = resolve_compute_precision(io_precisions, compute_type)

    fft_shader = make_fft_shader(
        tuple(buffer_shape),
        axis,
        inverse=inverse,
        normalize_inverse=normalize_inverse,
        r2c=r2c,
        input_map=input_map,
        output_map=output_map,
        input_type=resolved_input_type,
        output_type=resolved_output_type,
        compute_type=resolved_compute_type,
        input_signal_range=input_signal_range)

    if print_shader:
        print(fft_shader)

    fft_shader(*buffers, graph=graph)

def fft2(
    buffer: vd.Buffer,
    graph: vd.CommandGraph = None,
    print_shader: bool = False,
    input_map: vd.MappingFunction = None,
    output_map: vd.MappingFunction = None,
    output_type: vd.dtype = None,
    input_type: vd.dtype = None,
    compute_type: vd.dtype = None,
):
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    fft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=len(buffer.shape) - 2,
        input_map=input_map,
        output_type=output_type,
        input_type=input_type,
        compute_type=compute_type,
    )
    fft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=len(buffer.shape) - 1,
        output_map=output_map,
        output_type=output_type if output_map is None else None,
        input_type=input_type if output_map is None else None,
        compute_type=compute_type,
    )

def fft3(
    buffer: vd.Buffer,
    graph: vd.CommandGraph = None,
    print_shader: bool = False,
    input_map: vd.MappingFunction = None,
    output_map: vd.MappingFunction = None,
    output_type: vd.dtype = None,
    input_type: vd.dtype = None,
    compute_type: vd.dtype = None,
):
    assert len(buffer.shape) == 3, 'Buffer must have 3 dimensions'

    fft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=0,
        input_map=input_map,
        output_type=output_type,
        input_type=input_type,
        compute_type=compute_type,
    )
    fft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=1,
        output_type=output_type,
        input_type=input_type,
        compute_type=compute_type,
    )
    fft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=2,
        output_map=output_map,
        output_type=output_type if output_map is None else None,
        input_type=input_type if output_map is None else None,
        compute_type=compute_type,
    )


def ifft(
        buffer: vd.Buffer,
        graph: vd.CommandGraph = None,
        print_shader: bool = False,
        axis: int = None,
        name: str = None,
        normalize: bool = True,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        output_type: vd.dtype = None,
        input_type: vd.dtype = None,
        compute_type: vd.dtype = None):
    fft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=axis,
        name=name,
        inverse=True,
        normalize_inverse=normalize,
        input_map=input_map,
        output_map=output_map,
        output_type=output_type,
        input_type=input_type,
        compute_type=compute_type,
    )

def ifft2(
    buffer: vd.Buffer,
    graph: vd.CommandGraph = None,
    print_shader: bool = False,
    normalize: bool = True,
    input_map: vd.MappingFunction = None,
    output_map: vd.MappingFunction = None,
    output_type: vd.dtype = None,
    input_type: vd.dtype = None,
    compute_type: vd.dtype = None,
):
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    ifft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=len(buffer.shape) - 2,
        normalize=normalize,
        input_map=input_map,
        output_type=output_type,
        input_type=input_type,
        compute_type=compute_type,
    )
    ifft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=len(buffer.shape) - 1,
        normalize=normalize,
        output_map=output_map,
        output_type=output_type if output_map is None else None,
        input_type=input_type if output_map is None else None,
        compute_type=compute_type,
    )

def ifft3(
    buffer: vd.Buffer,
    graph: vd.CommandGraph = None,
    print_shader: bool = False,
    normalize: bool = True,
    input_map: vd.MappingFunction = None,
    output_map: vd.MappingFunction = None,
    output_type: vd.dtype = None,
    input_type: vd.dtype = None,
    compute_type: vd.dtype = None,
):
    assert len(buffer.shape) == 3, 'Buffer must have 3 dimensions'

    ifft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=0,
        normalize=normalize,
        input_map=input_map,
        output_type=output_type,
        input_type=input_type,
        compute_type=compute_type,
    )
    ifft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=1,
        normalize=normalize,
        output_type=output_type,
        input_type=input_type,
        compute_type=compute_type,
    )
    ifft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=2,
        normalize=normalize,
        output_map=output_map,
        output_type=output_type if output_map is None else None,
        input_type=input_type if output_map is None else None,
        compute_type=compute_type,
    )


def rfft(
    buffer: vd.RFFTBuffer,
    graph: vd.CommandGraph = None,
    print_shader: bool = False,
    name: str = None,
    compute_type: vd.dtype = None,
):
    fft(
        buffer,
        buffer_shape=buffer.real_shape,
        graph=graph,
        print_shader=print_shader,
        name=name,
        r2c=True,
        output_type=buffer.var_type,
        input_type=buffer.var_type,
        compute_type=compute_type,
    )

def rfft2(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False, compute_type: vd.dtype = None):
    assert len(buffer.real_shape) == 2 or len(buffer.real_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    rfft(buffer, graph=graph, print_shader=print_shader, compute_type=compute_type)
    fft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=len(buffer.real_shape) - 2,
        output_type=buffer.var_type,
        input_type=buffer.var_type,
        compute_type=compute_type,
    )

def rfft3(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False, compute_type: vd.dtype = None):
    assert len(buffer.real_shape) == 3, 'Buffer must have 3 dimensions'

    rfft(buffer, graph=graph, print_shader=print_shader, compute_type=compute_type)
    fft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=1,
        output_type=buffer.var_type,
        input_type=buffer.var_type,
        compute_type=compute_type,
    )
    fft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=0,
        output_type=buffer.var_type,
        input_type=buffer.var_type,
        compute_type=compute_type,
    )

def irfft(
    buffer: vd.RFFTBuffer,
    graph: vd.CommandGraph = None,
    print_shader: bool = False,
    name: str = None,
    normalize: bool = True,
    compute_type: vd.dtype = None,
):
    fft(
        buffer,
        buffer_shape=buffer.real_shape,
        graph=graph,
        print_shader=print_shader,
        name=name,
        inverse=True,
        normalize_inverse=normalize,
        r2c=True,
        output_type=buffer.var_type,
        input_type=buffer.var_type,
        compute_type=compute_type,
    )

def irfft2(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False, normalize: bool = True, compute_type: vd.dtype = None):
    assert len(buffer.real_shape) == 2 or len(buffer.real_shape) == 3, 'Buffer must have 2 or 3 dimensions'

    ifft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=len(buffer.real_shape) - 2,
        normalize=normalize,
        output_type=buffer.var_type,
        input_type=buffer.var_type,
        compute_type=compute_type,
    )
    irfft(buffer, graph=graph, print_shader=print_shader, normalize=normalize, compute_type=compute_type)

def irfft3(buffer: vd.RFFTBuffer, graph: vd.CommandGraph = None, print_shader: bool = False, normalize: bool = True, compute_type: vd.dtype = None):
    assert len(buffer.real_shape) == 3, 'Buffer must have 3 dimensions'

    ifft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=0,
        normalize=normalize,
        output_type=buffer.var_type,
        input_type=buffer.var_type,
        compute_type=compute_type,
    )
    ifft(
        buffer,
        graph=graph,
        print_shader=print_shader,
        axis=1,
        normalize=normalize,
        output_type=buffer.var_type,
        input_type=buffer.var_type,
        compute_type=compute_type,
    )
    irfft(buffer, graph=graph, print_shader=print_shader, normalize=normalize, compute_type=compute_type)

def convolve(
        *buffers: vd.Buffer,
        kernel_map: vd.MappingFunction = None,
        kernel_num: int = 1,
        buffer_shape: Tuple = None,
        graph: vd.CommandGraph = None,
        print_shader: bool = False,
        axis: int = None,
        normalize: bool = True,
        name: str = None,
        transposed_kernel: bool = False,
        kernel_inner_only: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        output_type: vd.dtype = None,
        input_type: vd.dtype = None,
        kernel_type: vd.dtype = None,
        compute_type: vd.dtype = None,
        input_signal_range: Union[Tuple[Optional[int], Optional[int]], None] = None):
    assert len(buffers) >= 1, "At least one buffer must be provided"

    if kernel_map is None and len(buffers) < 2:
        raise ValueError("convolve() requires at least an output buffer and kernel buffer")

    if buffer_shape is None:
        buffer_shape = buffers[0].shape

    resolved_output_type = _resolve_output_precision(buffers, output_map, output_type)
    resolved_input_type = _resolve_input_precision(input_map, output_map, input_type, resolved_output_type)
    resolved_kernel_type = _resolve_kernel_precision(buffers, kernel_map, kernel_type)

    io_precisions: List[vd.dtype] = []

    if output_map is None:
        io_precisions.append(resolved_output_type)
    else:
        _validate_map_argument_annotations(output_map, "output_map")

    if input_map is None:
        if resolved_input_type is not None:
            io_precisions.append(resolved_input_type)
    else:
        _validate_map_argument_annotations(input_map, "input_map")

    if kernel_map is None:
        io_precisions.append(resolved_kernel_type)
    else:
        _validate_map_argument_annotations(kernel_map, "kernel_map")

    resolved_compute_type = resolve_compute_precision(io_precisions, compute_type)

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
        input_type=resolved_input_type,
        output_type=resolved_output_type,
        kernel_type=resolved_kernel_type,
        compute_type=resolved_compute_type,
        input_signal_range=input_signal_range)

    if print_shader:
        print(fft_shader)

    fft_shader(*buffers, graph=graph)

def convolve2D(
        buffer: vd.Buffer,
        kernel: vd.Buffer,
        kernel_map: vd.MappingFunction = None,
        buffer_shape: Tuple = None,
        graph: vd.CommandGraph = None,
        print_shader: bool = False,
        normalize: bool = True,
        transposed_kernel: bool = False,
        kernel_inner_only: bool = False,
        input_map: vd.MappingFunction = None,
        output_map: vd.MappingFunction = None,
        output_type: vd.dtype = None,
        input_type: vd.dtype = None,
        kernel_type: vd.dtype = None,
        compute_type: vd.dtype = None):

    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    input_buffers = [buffer]

    if input_map is not None:
        input_buffers.append(buffer)

    output_buffers = [buffer]
    if output_map is not None:
        output_buffers.append(buffer)

    fft(
        *input_buffers,
        graph=graph,
        print_shader=print_shader,
        input_map=input_map,
        output_type=output_type,
        input_type=input_type,
        compute_type=compute_type,
    )
    convolve(
        buffer,
        kernel,
        kernel_map=kernel_map,
        buffer_shape=buffer_shape,
        graph=graph,
        transposed_kernel=transposed_kernel,
        kernel_inner_only=kernel_inner_only,
        print_shader=print_shader,
        axis=len(buffer.shape) - 2,
        normalize=normalize,
        output_type=output_type,
        input_type=input_type,
        kernel_type=kernel_type,
        compute_type=compute_type,
    )
    ifft(
        *output_buffers,
        graph=graph,
        print_shader=print_shader,
        normalize=normalize,
        output_map=output_map,
        output_type=output_type if output_map is None else None,
        input_type=input_type if output_map is None else None,
        compute_type=compute_type,
    )

def convolve2DR(
        buffer: vd.RFFTBuffer,
        kernel: vd.RFFTBuffer,
        kernel_map: vd.MappingFunction = None,
        buffer_shape: Tuple = None,
        transposed_kernel: bool = False,
        kernel_inner_only: bool = False,
        graph: vd.CommandGraph = None,
        print_shader: bool = False,
        normalize: bool = True,
        compute_type: vd.dtype = None):
    
    assert len(buffer.shape) == 2 or len(buffer.shape) == 3, 'Buffer must have 2 or 3 dimensions'

    rfft(buffer, graph=graph, print_shader=print_shader, compute_type=compute_type)
    convolve(
        buffer,
        kernel,
        kernel_map=kernel_map,
        buffer_shape=buffer_shape,
        graph=graph,
        transposed_kernel=transposed_kernel,
        kernel_inner_only=kernel_inner_only,
        print_shader=print_shader,
        axis=len(buffer.shape) - 2,
        normalize=normalize,
        output_type=buffer.var_type,
        input_type=buffer.var_type,
        kernel_type=kernel.var_type,
        compute_type=compute_type,
    )
    irfft(buffer, graph=graph, print_shader=print_shader, normalize=normalize, compute_type=compute_type)

def transpose(
        in_buffer: vd.Buffer,
        conv_shape: Tuple = None,
        axis: int = None,
        out_buffer: vd.Buffer = None,
        graph: vd.CommandGraph = None,
        kernel_inner_only: bool = False,
        print_shader: bool = False,
        input_type: vd.dtype = None,
        output_type: vd.dtype = None,
        compute_type: vd.dtype = None) -> vd.Buffer:

    resolved_input_type = in_buffer.var_type if input_type is None else input_type
    validate_complex_precision(resolved_input_type, arg_name="input_type")
    ensure_supported_complex_precision(resolved_input_type, role="Input")

    resolved_output_type = (
        out_buffer.var_type if (out_buffer is not None and output_type is None)
        else in_buffer.var_type if output_type is None
        else output_type
    )
    validate_complex_precision(resolved_output_type, arg_name="output_type")
    ensure_supported_complex_precision(resolved_output_type, role="Output")

    resolved_compute_type = resolve_compute_precision(
        [resolved_input_type, resolved_output_type],
        compute_type,
    )

    transposed_size = get_transposed_size(
        tuple(in_buffer.shape),
        axis=axis,
        compute_type=resolved_compute_type,
    )

    if out_buffer is None:
        out_buffer = vd.Buffer((transposed_size,), var_type=resolved_output_type)
    else:
        if out_buffer.var_type != resolved_output_type:
            raise ValueError(
                f"out_buffer type ({out_buffer.var_type.name}) does not match output_type ({resolved_output_type.name})"
            )

    assert out_buffer.size >= transposed_size, f"Output buffer size {out_buffer.size} does not match expected transposed size {transposed_size}"

    if conv_shape is None:
        conv_shape = in_buffer.shape

    transpose_shader = make_transpose_shader(
        tuple(conv_shape),
        axis=axis,
        kernel_inner_only=kernel_inner_only,
        input_type=resolved_input_type,
        output_type=resolved_output_type,
        compute_type=resolved_compute_type,
    )

    if print_shader:
        print(transpose_shader)

    transpose_shader(out_buffer, in_buffer, graph=graph)

    return out_buffer
