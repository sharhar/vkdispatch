import vkdispatch as vd
import vkdispatch_native

class descriptor_set:
    def __init__(self, plan: vd.compute_plan) -> None:
        self._handle = vkdispatch_native.descriptor_set_create(plan._handle)
    
    def bind_buffer(self, buffer: vd.buffer, binding: int) -> None:
        vkdispatch_native.descriptor_set_write_buffer(self._handle, binding, buffer._handle)
    