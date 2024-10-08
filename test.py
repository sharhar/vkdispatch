import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import vkdispatch_native

import numpy as np

vd.initialize(log_level=vd.LogLevel.INFO)

vd.make_context(devices=[0], queue_families=[[2]])

buff = vd.asbuffer(np.arange(10, dtype=np.float32))

cmd_list = vd.CommandList()

@vc.shader(exec_size=lambda args: args.buff.size)
def add_five(buff: Buff[f32]):
    tid = vc.global_invocation.x
    buff[tid] += 5


add_five(buff, cmd_list=cmd_list)

vkdispatch_native.record_conditional(cmd_list._handle)

add_five(buff, cmd_list=cmd_list)

vkdispatch_native.record_conditional_end(cmd_list._handle)

zeroed_byte = bytes([1])

print(buff.read(0))
cmd_list.submit(data=zeroed_byte)
print(buff.read(0))