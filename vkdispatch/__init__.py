from typing import Any
from .errors import check_for_errors
from .init import DeviceInfo
from .init import LogLevel
from .init import get_devices
from .init import initialize
from .buffer import asbuffer
from .buffer import Buffer
from .buffer import BufferKernelArgument
from .command_list import CommandList
from .command_list import get_command_list
from .command_list import get_command_list_handle
from .context import get_context
from .context import get_context_handle
from .context import make_context
from .descriptor_set import DescriptorSet
from .dtype import complex64
from .dtype import dtype
from .dtype import dtype_structure
from .dtype import float32
from .dtype import from_numpy_dtype
from .dtype import int32
from .dtype import ivec2
from .dtype import ivec4
from .dtype import mat2
from .dtype import mat4
from .dtype import to_numpy_dtype
from .dtype import uint32
from .dtype import uvec2
from .dtype import uvec4
from .dtype import vec2
from .dtype import vec4
from .image import image_format
from .image import image_type
from .image import image_view_type
from .image import Image
from .image import Image2D
from .image import Image2DArray
from .image import Image3D
from .shader_variable import ShaderVariable
from .shader_builder import BufferStructureProxy
from .shader_builder import shader
from .shader_builder import ShaderBuilder
from .stage_compute import ComputePlan
from .shader_decorator import compute_shader
from .shader_decorator import ShaderDispatcher
from .kernel_decorator import kernel
from .reductions import make_reduction
from .reductions import map_reduce
from .stage_fft import fft
from .stage_fft import FFTPlan
from .stage_fft import ifft
from .stage_fft import reset_fft_plans
#from .stage_transfer import stage_transfer_copy_buffer_to_image
from .stage_transfer import stage_transfer_copy_buffers
#from .stage_transfer import stage_transfer_copy_image
#from .stage_transfer import stage_transfer_copy_image_to_buffer

class Constant(ShaderVariable):
    def __init__(self) -> None:
        pass

class Variable(ShaderVariable):
    def __init__(self) -> None:
        pass