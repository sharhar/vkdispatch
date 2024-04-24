import vkdispatch as vd
import vkdispatch_native

import numpy as np

from typing import Callable, Any

class command_list:
    def __init__(self) -> None:
        self._handle: int = vkdispatch_native.command_list_create(vd.get_context_handle())
    
    def __del__(self) -> None:
        pass #vkdispatch_native.command_list_destroy(self._handle)

    def get_instance_size(self) -> int:
        return vkdispatch_native.command_list_get_instance_size(self._handle)

    def dispatch_shader(self, build_func: Callable[['vd.shader_builder', Any], None], blocks: tuple[int, int, int], local_size: tuple[int, int, int], static_args: list[vd.buffer | vd.image] = []) -> None:
        plan = vd.build_compute_plan(build_func, local_size, static_args)
        self.dispatch(plan, blocks)
    
    def dispatch(self, plan: 'vd.compute_plan', blocks: tuple[int, int, int]) -> None:
        plan.record(self, blocks)
    
    def submit(self, data: np.ndarray = None, instance_count: int = 1, device_index: int = 0) -> None:
        if self.get_instance_size() > 0 and data is not None:
            if data.size * np.dtype(data.dtype).itemsize != self.get_instance_size() * instance_count:
                raise ValueError("Numpy buffer sizes must match!")
        
        if data is None:
            data = np.zeros((instance_count, 1), dtype=np.uint8)

        vkdispatch_native.command_list_submit(self._handle, np.ascontiguousarray(data), instance_count, device_index)