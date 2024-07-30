from typing import Tuple

import vkdispatch as vd
import vkdispatch_native
from .command_list import CommandList
from .descriptor_set import DescriptorSet


class ComputePlan:
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
