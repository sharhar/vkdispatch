#import vkdispatch as vd
import vkdispatch.codegen as vc
#from vkdispatch.codegen.abreviations import *

import numpy as np

import sys

my_func = lambda x: x + 5

print(my_func(x = 5))

obj = (vc.Buffer[vc.f32], vc.Const[vc.i32])

if sys.argv[1] == "1":
    obj = (vc.Image1D[vc.f32], vc.Variable[vc.i32])

print(obj)

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