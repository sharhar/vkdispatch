import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

#vd.initialize(log_level=vd.LogLevel.VERBOSE, debug_mode=True)

vd.make_context(devices=[0], queue_families=[[2]])

buff = vd.asbuffer(np.arange(10, dtype=np.float32))

cmd_stream = vd.CommandStream()

@vd.shader(exec_size=lambda args: args.buff.size)
def add_five(buff: Buff[f32], value: Var[f32], value2: Var[f32]):
    tid = vc.global_invocation.x
    #vc.print(tid, value)
    buff[tid] += value
    buff[tid] += value2

launch_vars = vd.LaunchVariables(cmd_stream)

add_five(buff, launch_vars.new("val"), launch_vars.new("val2"), cmd_stream=cmd_stream)

def callback():
    launch_vars.set("val", [0.1, 0.02, 0.003, 0.0004])
    launch_vars.set("val2", [10, 200, 3000, 40000])

#callback = lambda: launch_vars.set("val", np.array([1, 2, 3, 4]).reshape(4, 1))

print(buff.read(0))
cmd_stream.submit(instance_count=4, callback=callback)
print(buff.read(0))