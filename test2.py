import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

@vd.shader()
def demo_shader(buff: Buff[f32]):
    tid = vc.global_invocation().x

    my_tid = tid + vc.global_invocation().y

    #print(my_tid)

    #my_tid[:] = my_tid + 4

    buff[tid] = tid * 2.0

print(demo_shader)