import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

vd.initialize(debug_mode=True)

@vd.shader("buff.size") #, flags=vc.ShaderFlags.NO_EXEC_BOUNDS)
def add_scalar(buff: Buff[f32], bias: Const[f32]):
    tid = vc.global_invocation_id().x
    vc.print("tid:", tid, "\\n")
    buff[tid] = buff[tid] + bias

buff = vd.buffer_f32(4)

add_scalar(buff, 1.0)

print(buff.read(0))

#print(add_scalar)