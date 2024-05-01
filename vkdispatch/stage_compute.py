import vkdispatch as vd
import vkdispatch_native

from typing import Callable, Any

import numpy as np

class compute_plan:
    def __init__(self, shader_source: str, binding_count: int, pc_size: int) -> None:
        
        self.binding_count = binding_count
        self.pc_size = pc_size
        self.shader_source = shader_source

        #for ii, line in enumerate(shader_source.split("\n")):
        #    print(f"{ii + 1:03d} | {line}")

        self._handle = vkdispatch_native.stage_compute_plan_create(vd.get_context_handle(), shader_source.encode(), binding_count, pc_size)
    
    def record(self, command_list: 'vd.command_list', desciptor_set: 'vd.descriptor_set', blocks: tuple[int, int, int]) -> None:
        vkdispatch_native.stage_compute_record(command_list._handle, self._handle, desciptor_set._handle, blocks[0], blocks[1], blocks[2])