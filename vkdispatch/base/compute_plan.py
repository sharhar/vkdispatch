from typing import Tuple

import vkdispatch_native

from .context import get_context_handle
from .errors import check_for_compute_stage_errors, check_for_errors


class ComputePlan:
    """
    ComputePlan is a wrapper for the native functions which create and dispatch Vulkan compute shaders.
    
    Attributes:
        pc_size (int): The size of the push constants for the compute shader (in bytes)
        shader_source (str): The source code of the compute shader (in GLSL)
        binding_list (list): A list of binding types for the shader resources.
        _handle (int): A pointer to the compute plan created by the native Vulkan dispatch.
    """

    def __init__(self, shader_source: str, binding_type_list: list, pc_size: int, shader_name: str) -> None:
        self.pc_size = pc_size
        self.shader_source = shader_source
        self.binding_list = binding_type_list

        self._handle = vkdispatch_native.stage_compute_plan_create(
            get_context_handle(), shader_source.encode(), self.binding_list, pc_size, shader_name.encode()
        )
        check_for_compute_stage_errors()
