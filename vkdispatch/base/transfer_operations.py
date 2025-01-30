import typing

import numpy as np

import vkdispatch_native

from .errors import check_for_errors
from .buffer import Buffer
from .command_list import CommandList

def stage_transfer_copy_buffers(
    command_list: CommandList,
    src: Buffer,
    dst: Buffer,
    size: typing.Optional[int] = None,
    src_offset: int = 0,
    dst_offset: int = 0,
) -> None:
    if size is None:
        assert (
            src.mem_size == dst.mem_size
        ), "Buffer memory sizes must match if size is None!"
        size = src.mem_size

    assert src_offset >= 0, "Src offset must be positive!"
    assert dst_offset >= 0, "Dst offset must be positive!"

    assert size + src_offset <= src.mem_size, "Src offset + size > src buffer size!"
    assert size + dst_offset <= dst.mem_size, "Dst offset + size > dst buffer size!"

    vkdispatch_native.stage_transfer_record_copy_buffer(
        command_list._handle, src._handle, dst._handle, src_offset, dst_offset, size
    )
    check_for_errors()

"""

def stage_transfer_copy_image(
    command_list: vkdispatch.CommandList,
    src: vkdispatch.image,
    dst: vkdispatch.image,
    extent: typing.Tuple[int, int, int] = None,
    src_offset: typing.Tuple[int, int, int] = (0, 0, 0),
    dst_offset: typing.Tuple[int, int, int] = (0, 0, 0),
    src_baseLayer: int = 0,
    src_layerCount: int = 1,
    dst_baseLayer: int = 0,
    dst_layerCount: int = 1,
) -> None:

    if extent is None:
        assert src.extent == dst.extent, "Image extents must match if extent is None!"

        extent = src.extent

    assert src_offset[0] >= 0, "Src offset x must be positive!"
    assert src_offset[1] >= 0, "Src offset y must be positive!"
    assert src_offset[2] >= 0, "Src offset z must be positive!"

    assert dst_offset[0] >= 0, "Dst offset x must be positive!"
    assert dst_offset[1] >= 0, "Dst offset y must be positive!"
    assert dst_offset[2] >= 0, "Dst offset z must be positive!"

    assert (
        src_offset[0] + extent[0] <= src.extent[0]
    ), "Src offset x + width > src width!"
    assert (
        src_offset[1] + extent[1] <= src.extent[1]
    ), "Src offset y + height > src height!"
    assert (
        src_offset[2] + extent[2] <= src.extent[2]
    ), "Src offset z + depth > src depth!"

    assert (
        dst_offset[0] + extent[0] <= dst.extent[0]
    ), "Dst offset x + width > dst width!"
    assert (
        dst_offset[1] + extent[1] <= dst.extent[1]
    ), "Dst offset y + height > dst height!"
    assert (
        dst_offset[2] + extent[2] <= dst.extent[2]
    ), "Dst offset z + depth > dst depth!"

    assert src_baseLayer >= 0, "Src base layer must not be negative!"
    assert dst_baseLayer >= 0, "Dst base layer must not be negative!"

    assert src_layerCount >= 1, "Src layer count must be at least 1!"
    assert dst_layerCount >= 1, "Dst layer count must be at least 1!"

    assert (
        src.layers >= src_baseLayer + src_layerCount
    ), "Src base layer + layer count > src layers!"
    assert (
        dst.layers >= dst_baseLayer + dst_layerCount
    ), "Dst base layer + layer count > dst layers!"

    vkdispatch_native.stage_transfer_record_copy_image(
        command_list._handle,
        src._handle,
        dst._handle,
        src_offset,
        dst_offset,
        extent,
        src_baseLayer,
        src_layerCount,
        dst_baseLayer,
        dst_layerCount,
    )


def stage_transfer_copy_buffer_to_image(
    command_list: vkdispatch.CommandList,
    image: vkdispatch.image,
    buffer: vkdispatch.Buffer,
    extent: typing.Tuple[int, int, int] = None,
    image_offset: typing.Tuple[int, int, int] = (0, 0, 0),
    buffer_offset: int = 0,
    buffer_row_length: int = 0,
    buffer_image_height: int = 0,
    image_baseLayer: int = 0,
    image_layerCount: int = 1,
) -> None:
    assert image_offset[0] >= 0, "Image offset x must be positive!"
    assert image_offset[1] >= 0, "Image offset y must be positive!"
    assert image_offset[2] >= 0, "Image offset z must be positive!"

    assert buffer_offset >= 0, "Buffer offset must be positive!"
    assert buffer_row_length >= 0, "Buffer row length must be positive!"
    assert buffer_image_height >= 0, "Buffer image height must be positive!"

    assert image_baseLayer >= 0, "Image base layer must not be negative!"
    assert image_layerCount >= 1, "Image layer count must be at least 1!"
    assert (
        image.layers >= image_baseLayer + image_layerCount
    ), "Image base layer + layer count > image layers!"

    if extent is None:
        assert (
            image.extent * image.block_size * image_layerCount <= buffer.mem_size
        ), "Image extent * layerCount * blockSize > Buffer mem_size!"

        extent = image.extent

    if buffer_row_length > 0 and buffer_image_height > 0:
        assert (
            buffer_offset + buffer_row_length * buffer_image_height * extent[2]
            >= buffer.mem_size
        ), "Buffer offset + row length * image height > buffer size!"
    elif buffer_row_length > 0:
        assert (
            buffer_offset + buffer_row_length * extent[1] * extent[2] >= buffer.mem_size
        ), "Buffer offset + row length * height > buffer size!"

    assert (
        image_offset[0] + extent[0] <= image.extent[0]
    ), "Image offset x + width > image width!"
    assert (
        image_offset[1] + extent[1] <= image.extent[1]
    ), "Image offset y + height > image height!"
    assert (
        image_offset[2] + extent[2] <= image.extent[2]
    ), "Image offset z + depth > image depth!"

    vkdispatch_native.stage_transfer_record_copy_buffer_to_image(
        command_list._handle,
        image._handle,
        buffer._handle,
        image_offset,
        buffer_offset,
        buffer_row_length,
        buffer_image_height,
        extent,
        image_baseLayer,
        image_layerCount,
    )


def stage_transfer_copy_image_to_buffer(
    command_list: vkdispatch.CommandList,
    image: vkdispatch.image,
    buffer: vkdispatch.Buffer,
    extent: typing.Tuple[int, int, int] = None,
    image_offset: typing.Tuple[int, int, int] = (0, 0, 0),
    buffer_offset: int = 0,
    buffer_row_length: int = 0,
    buffer_image_height: int = 0,
    image_baseLayer: int = 0,
    image_layerCount: int = 1,
) -> None:
    assert image_offset[0] >= 0, "Image offset x must be positive!"
    assert image_offset[1] >= 0, "Image offset y must be positive!"
    assert image_offset[2] >= 0, "Image offset z must be positive!"

    assert buffer_offset >= 0, "Buffer offset must be positive!"
    assert buffer_row_length >= 0, "Buffer row length must be positive!"
    assert buffer_image_height >= 0, "Buffer image height must be positive!"

    assert image_baseLayer >= 0, "Image base layer must not be negative!"
    assert image_layerCount >= 1, "Image layer count must be at least 1!"
    assert (
        image.layers >= image_baseLayer + image_layerCount
    ), "Image base layer + layer count > image layers!"

    if extent is None:
        assert (
            image.extent * image.block_size * image_layerCount <= buffer.mem_size
        ), "Image extent * layerCount * blockSize > Buffer mem_size!"

        extent = image.extent

    if buffer_row_length > 0 and buffer_image_height > 0:
        assert (
            buffer_offset + buffer_row_length * buffer_image_height * extent[2]
            >= buffer.mem_size
        ), "Buffer offset + row length * image height > buffer size!"
    elif buffer_row_length > 0:
        assert (
            buffer_offset + buffer_row_length * extent[1] * extent[2] >= buffer.mem_size
        ), "Buffer offset + row length * height > buffer size!"

    assert (
        image_offset[0] + extent[0] <= image.extent[0]
    ), "Image offset x + width > image width!"
    assert (
        image_offset[1] + extent[1] <= image.extent[1]
    ), "Image offset y + height > image height!"
    assert (
        image_offset[2] + extent[2] <= image.extent[2]
    ), "Image offset z + depth > image depth!"

    vkdispatch_native.stage_transfer_record_copy_image_to_buffer(
        command_list._handle,
        image._handle,
        buffer._handle,
        image_offset,
        buffer_offset,
        buffer_row_length,
        buffer_image_height,
        extent,
        image_baseLayer,
        image_layerCount,
    )

"""