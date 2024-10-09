#import vkdispatch as vd
#import vkdispatch.codegen as vc
#from vkdispatch.codegen.abreviations import *

import numpy as np

buffer = np.zeros((4, 16), dtype=np.uint8)

(buffer[:, 5:13]).view(np.int32)[:] = np.array([[19238, -1], [123098, 25687], [12301, 1234], [-102, -459145134]], dtype=np.int32)

print(buffer)

exit()

vd.initialize(log_level=vd.LogLevel.INFO)

vd.make_context(devices=[0], queue_families=[[2]])

buff = vd.asbuffer(np.arange(10, dtype=np.float32))

cmd_list = vd.CommandList()

@vc.shader(exec_size=lambda args: args.buff.size)
def add_five(buff: Buff[f32]):
    tid = vc.global_invocation.x
    buff[tid] += 5


add_five(buff, cmd_list=cmd_list)

conditional_index = cmd_list.record_conditional()

add_five(buff, cmd_list=cmd_list)

cmd_list.record_conditional_end()

zeroed_byte = bytes([0])

print(buff.read(0))
cmd_list.submit(data=zeroed_byte)
print(buff.read(0))