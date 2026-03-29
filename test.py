import os
os.environ["VKDISPATCH_BACKEND"] = "opencl"
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

vd.initialize(backend="opencl")

print("backend:", vd.get_backend())

buf3 = vd.Buffer((4,), vd.uvec3)

@vd.shader(4)
def fill3(buff: vc.Buff[vc.uv3]):
    tid = vc.global_invocation_id().x
    buff[tid] = vc.to_uvec3(tid, tid + 100, tid + 200)

print("=== uvec3 source ===")
print(fill3.get_src(line_numbers=True))
fill3(buf3)
print("uvec3 readback:")
print(buf3.read(0))

buf4 = vd.Buffer((4,), vd.uvec4)

@vd.shader(4)
def fill4(buff: vc.Buff[vc.uv4]):
    tid = vc.global_invocation_id().x
    buff[tid] = vc.to_uvec4(tid, tid + 100, tid + 200, tid + 300)

print("=== uvec4 source ===")
print(fill4.get_src(line_numbers=True))
fill4(buf4)
print("uvec4 readback:")
print(buf4.read(0))

shape = (1, 28, 22)
ravel_buf = vd.Buffer(shape, vd.uvec3)

@vd.shader("buff.size")
def ravel_test(buff: vc.Buff[vc.uv3]):
    ind = vc.global_invocation_id().x
    buff[ind] = vc.ravel_index(ind, buff.shape)

print("=== ravel source ===")
print(ravel_test.get_src(line_numbers=True))
ravel_test(ravel_buf)
out = ravel_buf.read(0)
print("ravel[(0, 5, 7)] =", tuple(out[(0, 5, 7)]))
