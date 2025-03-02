import numpy as np
import matplotlib.pyplot as plt
import vkdispatch as vd
import vkdispatch.codegen as vc

buff = vd.asbuffer(np.random.rand(10).astype(np.float32))

@vd.shader("buff.size")
def sum_map(buff: vc.Buff[vc.f32]) -> vc.f32:
    buff[vc.global_invocation().x] = 5

print(buff.read())

sum_map(buff)

print(buff.read())