import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

vd.initialize(backend="dummy")

vd.set_dummy_context_params(max_workgroup_size=(64, 1, 1))

@vd.shader("buff.size")
def add_scalar(buff: Buff[f32], bias: Const[f32]):
    tid = vc.global_invocation_id().x
    buff[tid] = buff[tid] + bias

buff = vd.buffer_f32(10)

add_scalar(buff, 1.0)

print(buff.read(0))

print(add_scalar)