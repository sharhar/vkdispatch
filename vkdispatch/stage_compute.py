import vkdispatch as vd
import vkdispatch_native

from typing import Callable, Any

import numpy as np

class compute_plan:
    def __init__(self, shader_source: str, binding_count: int, pc_size: int) -> None:
        
        self.binding_count = binding_count
        self.pc_size = pc_size
        self.shader_source = shader_source

        self._handle = vkdispatch_native.stage_compute_plan_create(vd.get_context_handle(), shader_source.encode(), binding_count, pc_size)
    
    def bind_buffer(self, buffer: vd.buffer, binding: int) -> None:
        vkdispatch_native.stage_compute_bind(self._handle, binding, buffer._handle)
    
    def record(self, command_list: 'vd.command_list', blocks: tuple[int, int, int]) -> None:
        vkdispatch_native.stage_compute_record(command_list._handle, self._handle, blocks[0], blocks[1], blocks[2])

def build_compute_plan(build_func: Callable[['vd.shader_builder', Any], None], local_size: tuple[int, int, int], static_args: list[vd.buffer | vd.image] = []) -> compute_plan:
    builder = vd.shader_builder()

    func_args = []

    for buff in static_args:
        if isinstance(buff, vd.buffer):
            func_args.append(builder.static_buffer(buff))
        else:
            raise ValueError("Only buffers are supported as static arguments!")

    if len(func_args) > 0:
        build_func(builder, *func_args)
    else:
        build_func(builder)

    plan = compute_plan(builder.build(local_size[0], local_size[1], local_size[2]), builder.binding_count, builder.pc_size)

    for binding in builder.bindings:
        plan.bind_buffer(binding[0], binding[1])

    return plan