import typing
from enum import Enum

import numpy as np

import vkdispatch
import vkdispatch_native


__MAPPING__ = {
    (np.uint8, 1),
    (np.uint8, 1),
    (np.uint8, 2),
    (np.uint8, 2),
    (np.uint8, 3),
    (np.uint8, 3),
    (np.uint8, 4),
    (np.uint8, 4),
}


class image_format(Enum):  # TODO: Fix class naming scheme to adhere to convention
    R8_UINT = 13
    R8_SINT = 14
    R8G8_UINT = 20
    R8G8_SINT = 21
    R8G8B8_UINT = 27
    R8G8B8_SINT = 28
    R8G8B8A8_UINT = 41
    R8G8B8A8_SINT = 42
    R16_UINT = 74
    R16_SINT = 75
    R16_SFLOAT = 76
    R16G16_UINT = 81
    R16G16_SINT = 82
    R16G16_SFLOAT = 83
    R16G16B16_UINT = 88
    R16G16B16_SINT = 89
    R16G16B16_SFLOAT = 90
    R16G16B16A16_UINT = 95
    R16G16B16A16_SINT = 96
    R16G16B16A16_SFLOAT = 97
    R32_UINT = 98
    R32_SINT = 99
    R32_SFLOAT = 100
    R32G32_UINT = 101
    R32G32_SINT = 102
    R32G32_SFLOAT = 103
    R32G32B32_UINT = 104
    R32G32B32_SINT = 105
    R32G32B32_SFLOAT = 106
    R32G32B32A32_UINT = 107
    R32G32B32A32_SINT = 108
    R32G32B32A32_SFLOAT = 109
    R64_UINT = 110
    R64_SINT = 111
    R64_SFLOAT = 112
    R64G64_UINT = 113
    R64G64_SINT = 114
    R64G64_SFLOAT = 115
    R64G64B64_UINT = 116
    R64G64B64_SINT = 117
    R64G64B64_SFLOAT = 118
    R64G64B64A64_UINT = 119
    R64G64B64A64_SINT = 120
    R64G64B64A64_SFLOAT = 121


# TODO: This can be moved into the enum class as an indexing method
def select_image_format(dtype: np.dtype, channels: int) -> image_format:
    assert channels in [1, 2, 3, 4], f"Unsupported number of channels ({channels})! Must be 1, 2, 3 or 4!"

    # NOTE: These large if-else statements can be better indexed and maintained by a
    # dictionary lookup scheme
    #
    # __MAPPING__ = {
    #     (np.uint8, 1): R8_UINT,
    #     (np.uint8, 2): R8G8_UINT,
    #     (np.uint8, 3): R8G8B8_UINT,
    #     ...
    # }
    # return __MAPPING__[(dtype, channels)]

    if dtype == np.uint8:
        if channels == 1:
            return image_format.R8_UINT
        elif channels == 2:
            return image_format.R8G8_UINT
        elif channels == 3:
            return image_format.R8G8B8_UINT
        elif channels == 4:
            return image_format.R8G8B8A8_UINT
    elif dtype == np.int8:
        if channels == 1:
            return image_format.R8_SINT
        elif channels == 2:
            return image_format.R8G8_SINT
        elif channels == 3:
            return image_format.R8G8B8_SINT
        elif channels == 4:
            return image_format.R8G8B8A8_SINT
    elif dtype == np.uint16:
        if channels == 1:
            return image_format.R16_UINT
        elif channels == 2:
            return image_format.R16G16_UINT
        elif channels == 3:
            return image_format.R16G16B16_UINT
        elif channels == 4:
            return image_format.R16G16B16A16_UINT
    elif dtype == np.int16:
        if channels == 1:
            return image_format.R16_SINT
        elif channels == 2:
            return image_format.R16G16_SINT
        elif channels == 3:
            return image_format.R16G16B16_SINT
        elif channels == 4:
            return image_format.R16G16B16A16_SINT
    elif dtype == np.uint32:
        if channels == 1:
            return image_format.R32_UINT
        elif channels == 2:
            return image_format.R32G32_UINT
        elif channels == 3:
            return image_format.R32G32B32_UINT
        elif channels == 4:
            return image_format.R32G32B32A32_UINT
    elif dtype == np.int32:
        if channels == 1:
            return image_format.R32_SINT
        elif channels == 2:
            return image_format.R32G32_SINT
        elif channels == 3:
            return image_format.R32G32B32_SINT
        elif channels == 4:
            return image_format.R32G32B32A32_SINT
    elif dtype == np.float16:
        if channels == 1:
            return image_format.R16_SFLOAT
        elif channels == 2:
            return image_format.R16G16_SFLOAT
        elif channels == 3:
            return image_format.R16G16B16_SFLOAT
        elif channels == 4:
            return image_format.R16G16B16A16_SFLOAT
    elif dtype == np.float32:
        if channels == 1:
            return image_format.R32_SFLOAT
        elif channels == 2:
            return image_format.R32G32_SFLOAT
        elif channels == 3:
            return image_format.R32G32B32_SFLOAT
        elif channels == 4:
            return image_format.R32G32B32A32_SFLOAT
    elif dtype == np.float64:
        if channels == 1:
            return image_format.R64_SFLOAT
        elif channels == 2:
            return image_format.R64G64_SFLOAT
        elif channels == 3:
            return image_format.R64G64B64_SFLOAT
        elif channels == 4:
            return image_format.R64G64B64A64_SFLOAT
    else:
        raise ValueError(f"Unsupported dtype ({dtype})!")


class image_type(Enum):
    TYPE_2D = (1,)
    TYPE_3D = (2,)


class image_view_type(Enum):
    VIEW_TYPE_2D = (1,)
    VIEW_TYPE_3D = (2,)
    VIEW_TYPE_2D_ARRAY = (5,)


class image:
    def __init__(
        self,
        shape: typing.Tuple[int],
        layers: int,
        dtype: type,
        channels: int,
        view_type: image_view_type,
    ) -> None:
        assert len(shape) == 2 or len(shape) == 3, "Shape must be 2D or 3D!"

        assert type(shape[0]) == int, "Shape must be a tuple of integers!"
        assert type(shape[1]) == int, "Shape must be a tuple of integers!"

        if len(shape) == 3:
            assert type(shape[2]) == int, "Shape must be a tuple of integers!"

        assert type(dtype) == type, "Dtype must be a numpy dtype!"
        assert type(channels) == int, "Channels must be an integer!"

        self.type = (
            image_type.TYPE_3D
            if view_type == image_view_type.VIEW_TYPE_3D
            else image_type.TYPE_2D
        )
        self.view_type = view_type
        self.format: image_format = select_image_format(dtype, channels)
        self.dtype: type = dtype
        self.layers: int = layers
        self.channels: int = channels

        self.shape: typing.Tuple[int] = (
            (layers, shape[0], shape[1]) if layers > 1 else shape
        )
        self.extent: typing.Tuple[int] = (
            shape if len(shape) == 3 else (shape[0], shape[1], 1)
        )
        self.array_shape: typing.Tuple[int] = (*self.shape, channels)

        self.block_size: int = vkdispatch_native.image_format_block_size(
            self.format.value[0]
        )
        self.mem_size: int = np.prod(self.shape) * self.block_size

        self._handle: int = vkdispatch_native.image_create(
            vkdispatch.get_context_handle(),
            self.extent,
            self.layers,
            self.format.value[0],
            self.type.value[0],
            self.view_type.value[0],
        )

    def __del__(self) -> None:
        pass  # vkdispatch_native.buffer_destroy(self._handle)

    def write(self, data: np.ndarray, device_index: int = -1) -> None:
        if data.size * np.dtype(data.dtype).itemsize != self.mem_size:
            raise ValueError("Numpy buffer sizes must match!")
        vkdispatch_native.image_write(
            self._handle,
            np.ascontiguousarray(data),
            [0, 0, 0],
            self.extent,
            0,
            self.layers,
            device_index,
        )

    def read(self, device_index: int = -1) -> np.ndarray:
        result = np.ndarray(shape=self.array_shape, dtype=self.dtype)
        vkdispatch_native.image_read(
            self._handle, result, [0, 0, 0], self.extent, 0, self.layers, device_index
        )
        return result

    # TODO: Update the 'other' argument to reference a Image class
    def copy_to(self, other: "image", device_index: int = -1) -> None:
        if other.shape != self.shape or other.channels != self.channels:
            raise ValueError("Buffer memory sizes must match!")
        vkdispatch_native.image_copy(
            self._handle,
            other._handle,
            [0, 0, 0],
            0,
            self.layers,
            [0, 0, 0],
            0,
            self.layers,
            self.extent,
            device_index,
        )


class image2d(image):
    def __init__(
        self, shape: typing.Tuple[int], dtype: type = np.float32, channels: int = 1
    ) -> None:
        assert len(shape) == 2, "Shape must be 2D!"
        super().__init__(shape, 1, dtype, channels, image_view_type.VIEW_TYPE_2D)


class image2d_array(image):
    def __init__(
        self,
        shape: typing.Tuple[int],
        layers: int,
        dtype: type = np.float32,
        channels: int = 1,
    ) -> None:
        assert len(shape) == 2, "Shape must be 2D!"
        super().__init__(
            shape, layers, dtype, channels, image_view_type.VIEW_TYPE_2D_ARRAY
        )


class image3d(image):
    def __init__(
        self, shape: typing.Tuple[int], dtype: type = np.float32, channels: int = 1
    ) -> None:
        assert len(shape) == 3, "Shape must be 3D!"
        super().__init__(shape, 1, dtype, channels, image_view_type.VIEW_TYPE_3D)
