import vkdispatch
import vkdispatch_native

import numpy as np

class compute_plan:
    def __init__(self, shader_source: str, binding_count: int, pc_size: int) -> None:
        
        self.binding_count = binding_count
        self.pc_size = pc_size
        self.shader_source = shader_source

        self._handle = vkdispatch_native.stage_compute_plan_create(vkdispatch.get_context_handle(), shader_source.encode(), binding_count, pc_size)
    
    def bind_buffer(self, buffer: vkdispatch.buffer, binding: int) -> None:
        vkdispatch_native.stage_compute_bind(self._handle, binding, buffer._handle)
    
    def record(self, command_list: vkdispatch.command_list, blocks: tuple[int, int, int]) -> None:
        vkdispatch_native.stage_compute_record(command_list._handle, self._handle, blocks[0], blocks[1], blocks[2])