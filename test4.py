import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import torch

vd.initialize(backend="pycuda")

x = torch.randn(1024, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)

print(x)

bx = vd.from_cuda_array(x)
by = vd.from_cuda_array(y)

graph = vd.CommandGraph()
# record shader calls using bx/by...
# graph.set_var("scale", 2.0)

@vd.shader("buff.size")
def add_scalar(buff: Buff[f32], bias: Var[f32]):
    tid = vc.global_invocation_id().x
    buff[tid] = buff[tid] + bias

add_scalar(bx, graph.bind_var("scale"), graph=graph)

cap = graph.prepare_cuda_capture(instance_count=1)
graph.set_var("scale", 1.0)
graph.update_captured_args(cap)

g = torch.cuda.CUDAGraph()
stream = torch.cuda.current_stream()

with torch.cuda.graph(g):
    graph.submit(cuda_stream=stream, capture=cap)

# Later: change push constants / uniforms and replay
graph.set_var("scale", 3.0)
graph.update_captured_args(cap)
g.replay()

# print x tensor
print(x)

exit()

@vd.shader("buff.size") #, flags=vc.ShaderFlags.NO_EXEC_BOUNDS)
def add_scalar(buff: Buff[f32], bias: Const[f32]):
    tid = vc.global_invocation_id().x
    #vc.print("tid:", tid, "\\n")
    buff[tid] = buff[tid] + bias

buff = vd.buffer_f32(4)

add_scalar(buff, 1.0)

print(buff)

print(buff.read(0))

#print(add_scalar)