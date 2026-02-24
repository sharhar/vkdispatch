import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

vd.initialize(backend="pycuda")

dtp = f64

@vd.shader("buff.size")
def add_scalar(buff: Buff[dtp], bias: Const[dtp]):
    tid = vc.global_invocation_id().x
    buff[tid] = buff[tid] + bias

buff = vd.Buffer((4,), var_type=dtp)

add_scalar(buff, 1.12345678901234567890)

#print(buff)

print(buff.read(0))

#print(add_scalar)
