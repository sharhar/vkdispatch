import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

vc.set_codegen_backend("cuda")

@vd.shader("buff.size")
def add_scalar(buff: Buff[f32], bias: Const[f32]):
    tid = vc.global_invocation_id().x
    buff[tid] = buff[tid] + bias


vd.fft.cache_clear()

print(add_scalar)