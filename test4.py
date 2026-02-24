import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

#vd.initialize(backend="pycuda")

@vd.shader("buff.size")
def add_scalar(buff: Buff[f16], bias: Const[f16]):
    tid = vc.global_invocation_id().x
    buff[tid] = buff[tid] + bias

buff = vd.buffer_f16(4)

add_scalar(buff, 1.0)

#print(buff)

print(buff.read(0))

print(add_scalar)

