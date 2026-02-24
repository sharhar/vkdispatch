import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *
import numpy as np
np.set_printoptions(precision=18)
vd.initialize(backend="cuda-python")

dtp = v2

@vd.shader("buff.size")
def add_scalar(buff: Buff[dtp], bias: Const[dtp]):
    tid = vc.global_invocation_id().x
    buff[tid] = buff[tid] + vc.sin(bias)

buff = vd.Buffer((4,), var_type=dtp)

add_scalar(buff, (1.12345678901234567890, 2.12345678901234567890))

print(f"{float(buff.read(0)[0][0]), float(buff.read(0)[0][1])}")

#print(add_scalar)