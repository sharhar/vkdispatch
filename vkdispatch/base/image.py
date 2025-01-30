import typing
from enum import Enum

import numpy as np

import vkdispatch as vd
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
def select_image_format(dtype: vd.dtype, channels: int) -> image_format:
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

    """

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
    el """
    
    if dtype == vd.uint32:
        if channels == 1:
            return image_format.R32_UINT
        elif channels == 2:
            return image_format.R32G32_UINT
        elif channels == 3:
            return image_format.R32G32B32_UINT
        elif channels == 4:
            return image_format.R32G32B32A32_UINT
    elif dtype == vd.int32:
        if channels == 1:
            return image_format.R32_SINT
        elif channels == 2:
            return image_format.R32G32_SINT
        elif channels == 3:
            return image_format.R32G32B32_SINT
        elif channels == 4:
            return image_format.R32G32B32A32_SINT
    #elif dtype == np.float16:
    #    if channels == 1:
    #        return image_format.R16_SFLOAT
    #    elif channels == 2:
    #        return image_format.R16G16_SFLOAT
    #    elif channels == 3:
    #        return image_format.R16G16B16_SFLOAT
    #    elif channels == 4:
    #        return image_format.R16G16B16A16_SFLOAT
    elif dtype == vd.float32:
        if channels == 1:
            return image_format.R32_SFLOAT
        elif channels == 2:
            return image_format.R32G32_SFLOAT
        elif channels == 3:
            return image_format.R32G32B32_SFLOAT
        elif channels == 4:
            return image_format.R32G32B32A32_SFLOAT
    #elif dtype == vd.float64:
    #    if channels == 1:
    #        return image_format.R64_SFLOAT
    #    elif channels == 2:
    #        return image_format.R64G64_SFLOAT
    #    elif channels == 3:
    #        return image_format.R64G64B64_SFLOAT
    #    elif channels == 4:
    #        return image_format.R64G64B64A64_SFLOAT
    else:
        raise ValueError(f"Unsupported dtype ({dtype})!")


class image_type(Enum):
    TYPE_1D = (0,)
    TYPE_2D = (1,)
    TYPE_3D = (2,)


class image_view_type(Enum):
    VIEW_TYPE_1D = (0,)
    VIEW_TYPE_2D = (1,)
    VIEW_TYPE_3D = (2,)
    VIEW_TYPE_2D_ARRAY = (5,)

class Filter(Enum):
    NEAREST = 0
    LINEAR = 1

class AddressMode(Enum):
    REPEAT = 0
    MIRRORED_REPEAT = 1
    CLAMP_TO_EDGE = 2
    CLAMP_TO_BORDER = 3
    MIRROR_CLAMP_TO_EDGE = 4

class BorderColor(Enum):
    FLOAT_TRANSPARENT_BLACK = 0
    INT_TRANSPARENT_BLACK = 1
    FLOAT_OPAQUE_BLACK = 2
    INT_OPAQUE_BLACK = 3
    FLOAT_OPAQUE_WHITE = 4
    INT_OPAQUE_WHITE = 5

class Sampler:
    def __init__(self,
                    image: "Image",
                    mag_filter: Filter = Filter.LINEAR,
                    min_filter: Filter = Filter.LINEAR,
                    mip_filter: Filter = Filter.LINEAR,
                    address_mode: AddressMode = AddressMode.CLAMP_TO_EDGE,
                    mip_lod_bias: float = 0.0,
                    min_lod: float = 0.0,
                    max_lod: float = 0.0,
                    border_color: BorderColor = BorderColor.FLOAT_OPAQUE_WHITE
                ) -> None:
        self.image = image
        
        self._handle = vkdispatch_native.image_create_sampler(
            vd.get_context_handle(),
            mag_filter.value,
            min_filter.value,
            mip_filter.value,
            address_mode.value,
            mip_lod_bias,
            min_lod,
            max_lod,
            border_color.value
        )

class Image:
    def __init__(
        self,
        shape: typing.Tuple[int, ...],
        layers: int,
        dtype: type,
        channels: int,
        view_type: image_view_type,
        enable_mipmaps: bool = False,
    ) -> None:
        assert len(shape) == 1 or len(shape) == 2 or len(shape) == 3, "Shape must be 2D or 3D!"

        assert type(shape[0]) == int, "Shape must be a tuple of integers!"
        
        if len(shape) > 1:
            assert type(shape[1]) == int, "Shape must be a tuple of integers!"

        if len(shape) == 3:
            assert type(shape[2]) == int, "Shape must be a tuple of integers!"

        assert issubclass(dtype, vd.dtype), "Dtype must be a dtype!"
        assert type(channels) == int, "Channels must be an integer!"

        self.type = image_type.TYPE_1D

        if view_type == image_view_type.VIEW_TYPE_2D:
            self.type = image_type.TYPE_2D
        
        if view_type == image_view_type.VIEW_TYPE_3D:
            self.type = image_type.TYPE_3D
        
        self.view_type = view_type
        self.format: image_format = select_image_format(dtype, channels)
        self.dtype: vd.dtype = dtype
        self.layers: int = layers
        self.channels: int = channels

        self.shape: typing.Tuple[int] = shape

        if layers > 1:
            self.shape = (layers, *shape)

        self.extent: typing.Tuple[int] = shape

        if len(shape) == 1:
            self.extent = (shape[0], 1, 1)
        
        if len(shape) == 2:
            self.extent = (shape[0], shape[1], 1)

        self.array_shape: typing.Tuple[int] = (*self.shape, channels)

        if channels == 1:
            self.array_shape = self.array_shape[:-1]

        self.block_size: int = vkdispatch_native.image_format_block_size(
            self.format.value
        )

        self.mem_size: int = np.prod(self.shape) * self.block_size

        self._handle: int = vkdispatch_native.image_create(
            vd.get_context_handle(),
            self.extent,
            self.layers,
            self.format.value,
            self.type.value[0],
            self.view_type.value[0],
            1 if enable_mipmaps else 0,
        )

    def __del__(self) -> None:
        pass  # vkdispatch_native.buffer_destroy(self._handle)

    def write(self, data: np.ndarray, device_index: int = -1) -> None:
        if data.size * np.dtype(data.dtype).itemsize != self.mem_size:
            raise ValueError(f"Numpy buffer sizes must match! {data.size * np.dtype(data.dtype).itemsize} != {self.mem_size}")
        vkdispatch_native.image_write(
            self._handle,
            np.ascontiguousarray(data).tobytes(),
            [0, 0, 0],
            self.extent,
            0,
            self.layers,
            device_index,
        )

    def read(self, device_index: int = 0) -> np.ndarray:
        true_scalar = self.dtype.scalar

        if self.dtype.scalar is None:
            true_scalar = self.dtype

        out_size = np.prod(self.array_shape) * true_scalar.item_size
        out_bytes = vkdispatch_native.image_read(
            self._handle, out_size, [0, 0, 0], self.extent, 0, self.layers, device_index
        )
        return np.frombuffer(out_bytes, dtype=vd.to_numpy_dtype(true_scalar)).reshape(self.array_shape)
    
    def sample(self, 
                    mag_filter: Filter = Filter.LINEAR,
                    min_filter: Filter = Filter.LINEAR,
                    mip_filter: Filter = Filter.LINEAR,
                    address_mode: AddressMode = AddressMode.CLAMP_TO_EDGE,
                    mip_lod_bias: float = 0.0,
                    min_lod: float = 0.0,
                    max_lod: float = 0.0,
                    border_color: BorderColor = BorderColor.FLOAT_OPAQUE_WHITE
                ) -> Sampler:
        return Sampler(
            self,
            mag_filter,
            min_filter,
            mip_filter,
            address_mode,
            mip_lod_bias,
            min_lod,
            max_lod,
            border_color    
        )

class Image1D(Image):
    def __init__(self, shape: int, dtype: type, channels: int = 1, enable_mipmaps: bool = False) -> None:
        super().__init__((shape, ), 1, dtype, channels, image_view_type.VIEW_TYPE_1D, enable_mipmaps)


    @classmethod
    def __class_getitem__(cls, arg: vd.dtype) -> type:
        raise RuntimeError("Cannot index into vd.Image1D! Perhaps you meant to use vc.Image1D?")

class Image2D(Image):
    def __init__(
        self, shape: typing.Tuple[int, int], dtype: type = np.float32, channels: int = 1, enable_mipmaps: bool = False
    ) -> None:
        assert len(shape) == 2, "Shape must be 2D!"
        super().__init__(shape, 1, dtype, channels, image_view_type.VIEW_TYPE_2D, enable_mipmaps)
    
    @classmethod
    def __class_getitem__(cls, arg: vd.dtype) -> type:
        raise RuntimeError("Cannot index into vd.Image2D! Perhaps you meant to use vc.Image2D?")


class Image2DArray(Image):
    def __init__(
        self,
        shape: typing.Tuple[int, int],
        layers: int,
        dtype: type = np.float32,
        channels: int = 1,
        enable_mipmaps: bool = False
    ) -> None:
        assert len(shape) == 2, "Shape must be 2D!"
        super().__init__(
            shape, layers, dtype, channels, image_view_type.VIEW_TYPE_2D_ARRAY, enable_mipmaps
        )
    
    @classmethod
    def __class_getitem__(cls, arg: tuple) -> type:
        raise RuntimeError("Cannot index into vd.Image2DArray! Perhaps you meant to use vc.Image2DArray?")


class Image3D(Image):
    def __init__(
        self, shape: typing.Tuple[int, int, int], dtype: type = np.float32, channels: int = 1, enable_mipmaps: bool = False
    ) -> None:
        assert len(shape) == 3, "Shape must be 3D!"
        super().__init__(shape, 1, dtype, channels, image_view_type.VIEW_TYPE_3D, enable_mipmaps)
    
    @classmethod
    def __class_getitem__(cls, arg: vd.dtype) -> type:
        raise RuntimeError("Cannot index into vd.Image3D! Perhaps you meant to use vc.Image3D?")
