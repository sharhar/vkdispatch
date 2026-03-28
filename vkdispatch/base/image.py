import typing
from enum import Enum

from ..backends.backend_selection import native

from ..compat import numpy_compat as npc
from . import dtype as vdt
from .context import Handle

__MAPPING__ = set()


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
def select_image_format(dtype: vdt.dtype, channels: int) -> image_format:
    if channels < 1 or channels > 4:
        raise ValueError(f"Unsupported number of channels ({channels})! Must be 1, 2, 3 or 4!")

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

    if dtype == vdt.uint32:
        if channels == 1:
            return image_format.R32_UINT
        elif channels == 2:
            return image_format.R32G32_UINT
        elif channels == 3:
            return image_format.R32G32B32_UINT
        elif channels == 4:
            return image_format.R32G32B32A32_UINT
    elif dtype == vdt.int32:
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
    elif dtype == vdt.float32:
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
    """
    Defines the type of an image.

    Attributes:
        TYPE_1D (int): A 1-dimensional image.
        TYPE_2D (int): A 2-dimensional image.
        TYPE_3D (int): A 3-dimensional image.
    """
    TYPE_1D = (0,)
    TYPE_2D = (1,)
    TYPE_3D = (2,)


class image_view_type(Enum):
    """
    Defines the type of an image view.

    Attributes:
        VIEW_TYPE_1D (int): A 1-dimensional image view.
        VIEW_TYPE_2D (int): A 2-dimensional image view.
        VIEW_TYPE_3D (int): A 3-dimensional image view.
        VIEW_TYPE_2D_ARRAY (int): A 2D array of images.
    """
    VIEW_TYPE_1D = (0,)
    VIEW_TYPE_2D = (1,)
    VIEW_TYPE_3D = (2,)
    VIEW_TYPE_2D_ARRAY = (5,)


class Filter(Enum):
    """
    Defines the filter used for image sampling.

    Attributes:
        NEAREST (int): Nearest neighbor filtering.
        LINEAR (int): Linear interpolation filtering.
    """
    NEAREST = 0
    LINEAR = 1


class AddressMode(Enum):
    """
    Defines how to handle out-of-bounds addresses when accessing an image.

    Attributes:
        REPEAT (int): Repeat the image data when accessing out-of-bounds addresses.
        MIRRORED_REPEAT (int): Mirror and repeat the image data when accessing out-of-bounds addresses.
        CLAMP_TO_EDGE (int): Clamp out-of-bounds addresses to the edge of the image.
        CLAMP_TO_BORDER (int): Clamp out-of-bounds addresses to a specific border color.
        MIRROR_CLAMP_TO_EDGE (int): Mirror the image and clamp out-of-bounds addresses to the edge of the image.
    """
    REPEAT = 0
    MIRRORED_REPEAT = 1
    CLAMP_TO_EDGE = 2
    CLAMP_TO_BORDER = 3
    MIRROR_CLAMP_TO_EDGE = 4


class BorderColor(Enum):
    """
    Defines the border color used when clamping out-of-bounds addresses.

    Attributes:
        FLOAT_TRANSPARENT_BLACK (int): A fully transparent black border.
        INT_TRANSPARENT_BLACK (int): A fully transparent black border.
        FLOAT_OPAQUE_BLACK (int): An opaque black border with a specific alpha value.
        INT_OPAQUE_BLACK (int): An opaque black border with a specific alpha value.
        FLOAT_OPAQUE_WHITE (int): An opaque white border with a specific alpha value.
        INT_OPAQUE_WHITE (int): An opaque white border with a specific alpha value.
    """
    FLOAT_TRANSPARENT_BLACK = 0
    INT_TRANSPARENT_BLACK = 1
    FLOAT_OPAQUE_BLACK = 2
    INT_OPAQUE_BLACK = 3
    FLOAT_OPAQUE_WHITE = 4
    INT_OPAQUE_WHITE = 5

class Sampler(Handle):
    image: "Image"
    
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
        super().__init__()

        self.image = image
        
        handle = native.image_create_sampler(
            self.context._handle,
            mag_filter.value,
            min_filter.value,
            mip_filter.value,
            address_mode.value,
            mip_lod_bias,
            min_lod,
            max_lod,
            border_color.value
        )

        self.register_handle(handle)
        self.register_parent(image)

    def _destroy(self):
        native.image_destroy_sampler(self._handle)
    
    def __del__(self) -> None:
        self.destroy()

class Image(Handle):
    def __init__(
        self,
        shape: typing.Tuple[int, ...],
        layers: int,
        dtype: type,
        channels: int,
        view_type: image_view_type,
        enable_mipmaps: bool = False,
    ) -> None:
        super().__init__()

        if len(shape) < 1 or len(shape) > 3:
            raise ValueError("Shape must be 1D, 2D or 3D!")

        if type(shape[0]) != int:
            raise ValueError("Shape must be a tuple of integers!")
        
        if len(shape) > 1 and type(shape[1]) != int:
            raise ValueError("Shape must be a tuple of integers!")

        if len(shape) >2 and type(shape[2]) != int:
            raise ValueError("Shape must be a tuple of integers!")
        
        if not issubclass(dtype, vdt.dtype):
            raise ValueError("Dtype must be a dtype!")
        
        if type(channels) != int:
            raise ValueError("Channels must be an integer!")

        self.type = image_type.TYPE_1D

        if view_type == image_view_type.VIEW_TYPE_2D:
            self.type = image_type.TYPE_2D
        
        if view_type == image_view_type.VIEW_TYPE_3D:
            self.type = image_type.TYPE_3D
        
        self.view_type = view_type
        self.format: image_format = select_image_format(dtype, channels)
        self.dtype: vdt.dtype = dtype
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

        self.block_size: int = native.image_format_block_size(
            self.format.value
        )

        self.mem_size: int = npc.prod(self.shape) * self.block_size

        handle: int = native.image_create(
            self.context._handle,
            self.extent,
            self.layers,
            self.format.value,
            self.type.value[0],
            self.view_type.value[0],
            1 if enable_mipmaps else 0,
        )

        self.register_handle(handle)

    def _destroy(self) -> None:
        native.image_destroy(self._handle)

    def __del__(self) -> None:
        self.destroy()

    def write(self, data: typing.Any, device_index: int = -1) -> None:
        if npc.is_array_like(data):
            true_data = npc.as_contiguous_bytes(data)
            data_size = npc.array_nbytes(data)
        elif npc.is_bytes_like(data):
            true_data = npc.ensure_bytes(data)
            data_size = len(true_data)
        else:
            raise TypeError("Expected array-like or bytes-like image input")

        if data_size != self.mem_size:
            raise ValueError(f"Image buffer sizes must match! {data_size} != {self.mem_size}")

        native.image_write(
            self._handle,
            true_data,
            [0, 0, 0],
            self.extent,
            0,
            self.layers,
            device_index,
        )

    def read(self, device_index: int = 0):
        true_scalar = self.dtype.scalar

        if self.dtype.scalar is None:
            true_scalar = self.dtype

        out_size = npc.prod(self.array_shape) * true_scalar.item_size
        out_bytes = native.image_read(
            self._handle, out_size, [0, 0, 0], self.extent, 0, self.layers, device_index
        )
        return npc.from_buffer(out_bytes, dtype=vdt.to_numpy_dtype(true_scalar), shape=self.array_shape)
    
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
    def __class_getitem__(cls, arg: vdt.dtype) -> type:
        raise RuntimeError("Cannot index into vd.Image1D! Perhaps you meant to use vc.Image1D?")

class Image2D(Image):
    def __init__(
        self, shape: typing.Tuple[int, int], dtype: type = vdt.float32, channels: int = 1, enable_mipmaps: bool = False
    ) -> None:
        if len(shape) != 2:
            raise ValueError("Shape must be 2D!")
        super().__init__(shape, 1, dtype, channels, image_view_type.VIEW_TYPE_2D, enable_mipmaps)
    
    @classmethod
    def __class_getitem__(cls, arg: vdt.dtype) -> type:
        raise RuntimeError("Cannot index into vd.Image2D! Perhaps you meant to use vc.Image2D?")


class Image2DArray(Image):
    def __init__(
        self,
        shape: typing.Tuple[int, int],
        layers: int,
        dtype: type = vdt.float32,
        channels: int = 1,
        enable_mipmaps: bool = False
    ) -> None:
        if len(shape) != 2:
            raise ValueError("Shape must be 2D!")

        super().__init__(
            shape, layers, dtype, channels, image_view_type.VIEW_TYPE_2D_ARRAY, enable_mipmaps
        )
    
    @classmethod
    def __class_getitem__(cls, arg: tuple) -> type:
        raise RuntimeError("Cannot index into vd.Image2DArray! Perhaps you meant to use vc.Image2DArray?")


class Image3D(Image):
    def __init__(
        self, shape: typing.Tuple[int, int, int], dtype: type = vdt.float32, channels: int = 1, enable_mipmaps: bool = False
    ) -> None:
        if len(shape) != 3:
            raise ValueError("Shape must be 3D!")
        
        super().__init__(shape, 1, dtype, channels, image_view_type.VIEW_TYPE_3D, enable_mipmaps)
    
    @classmethod
    def __class_getitem__(cls, arg: vdt.dtype) -> type:
        raise RuntimeError("Cannot index into vd.Image3D! Perhaps you meant to use vc.Image3D?")
