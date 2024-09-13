from typing import Tuple

import vkdispatch as vd
import vkdispatch_native
from .command_list import CommandList
from .descriptor_set import DescriptorSet


class ComputePlan:
    """
    ComputePlan is a wrapper for the native functions which create and dispatch Vulkan compute shaders.
    Attributes:
        pc_size (int): The size of the push constants for the compute shader (in bytes)
        shader_source (str): The source code of the compute shader (in GLSL)
        binding_list (list): A list of binding types for the shader resources.
        _handle: A handle to the compute plan created by the native Vulkan dispatch.
    Functions:
        __init__(shader_source: str, binding_type_list: list, pc_size: int, shader_name: str) -> None:
            Initializes the ComputePlan with the given shader source, binding types, push constant size, and shader name.
        record(command_list: CommandList, descriptor_set: DescriptorSet, blocks: Tuple[int, int, int]) -> None:
            Records the compute shader execution into the provided command list using the specified descriptor set and block dimensions.
    """


    def __init__(self, shader_source: str, binding_type_list: list, pc_size: int, shader_name: str) -> None:
        self.pc_size = pc_size
        self.shader_source = shader_source
        self.binding_list = binding_type_list

        self._handle = vkdispatch_native.stage_compute_plan_create(
            vd.get_context_handle(), shader_source.encode(), self.binding_list, pc_size, shader_name.encode()
        )
        vd.check_for_compute_stage_errors()

    def record(
        self,
        command_list: CommandList,
        descriptor_set: DescriptorSet,
        blocks: Tuple[int, int, int],
    ) -> None:
        vkdispatch_native.stage_compute_record(
            command_list._handle,
            self._handle,
            descriptor_set._handle,
            blocks[0],
            blocks[1],
            blocks[2],
        )
        vd.check_for_errors()
