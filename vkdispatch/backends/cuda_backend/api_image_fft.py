from __future__ import annotations

from . import _state as state
from ._constants import _IMAGE_BLOCK_SIZES
from ._helpers import _set_error


def image_create(context, extent, layers, format, type, view_type, generate_mips):
    _ = context
    _ = extent
    _ = layers
    _ = format
    _ = type
    _ = view_type
    _ = generate_mips
    _set_error("CUDA Python backend does not support image objects yet")
    return 0


def image_destroy(image):
    state._images.pop(int(image), None)


def image_create_sampler(
    context,
    mag_filter,
    min_filter,
    mip_mode,
    address_mode,
    mip_lod_bias,
    min_lod,
    max_lod,
    border_color,
):
    _ = context
    _ = mag_filter
    _ = min_filter
    _ = mip_mode
    _ = address_mode
    _ = mip_lod_bias
    _ = min_lod
    _ = max_lod
    _ = border_color
    _set_error("CUDA Python backend does not support image samplers yet")
    return 0


def image_destroy_sampler(sampler):
    state._samplers.pop(int(sampler), None)


def image_write(image, data, offset, extent, baseLayer, layerCount, device_index):
    _ = image
    _ = data
    _ = offset
    _ = extent
    _ = baseLayer
    _ = layerCount
    _ = device_index
    _set_error("CUDA Python backend does not support image writes yet")


def image_format_block_size(format):
    return int(_IMAGE_BLOCK_SIZES.get(int(format), 4))


def image_read(image, out_size, offset, extent, baseLayer, layerCount, device_index):
    _ = image
    _ = offset
    _ = extent
    _ = baseLayer
    _ = layerCount
    _ = device_index
    _set_error("CUDA Python backend does not support image reads yet")
    return bytes(max(0, int(out_size)))


def stage_fft_plan_create(
    context,
    dims,
    axes,
    buffer_size,
    do_r2c,
    normalize,
    pad_left,
    pad_right,
    frequency_zeropadding,
    kernel_num,
    kernel_convolution,
    conjugate_convolution,
    convolution_features,
    input_buffer_size,
    num_batches,
    single_kernel_multiple_batches,
    keep_shader_code,
):
    _ = context
    _ = dims
    _ = axes
    _ = buffer_size
    _ = do_r2c
    _ = normalize
    _ = pad_left
    _ = pad_right
    _ = frequency_zeropadding
    _ = kernel_num
    _ = kernel_convolution
    _ = conjugate_convolution
    _ = convolution_features
    _ = input_buffer_size
    _ = num_batches
    _ = single_kernel_multiple_batches
    _ = keep_shader_code
    _set_error("CUDA Python backend does not support FFT plans yet")
    return 0


def stage_fft_plan_destroy(plan):
    state._fft_plans.pop(int(plan), None)


def stage_fft_record(command_list, plan, buffer, inverse, kernel, input_buffer):
    _ = command_list
    _ = plan
    _ = buffer
    _ = inverse
    _ = kernel
    _ = input_buffer
    _set_error("CUDA Python backend does not support FFT stages yet")
