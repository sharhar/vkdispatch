import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *
import numpy as np
np.set_printoptions(precision=18)
vd.initialize(backend="pycuda")

dtp = i16

@vd.shader("buff.size")
def add_scalar(buff: Buff[dtp], bias: Const[dtp]):
    tid = vc.global_invocation_id().x
    buff[tid] = buff[tid] + bias

buff = vd.Buffer((4,), var_type=dtp)

add_scalar(buff, 23452)

print(f"{buff.read(0)[0]}")

print(add_scalar)