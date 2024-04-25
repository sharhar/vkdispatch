import vkdispatch as vd 
import numpy as np

arrs = np.random.rand(4, 4).astype(np.float32)
arrs2 = np.random.rand(4, 4).astype(np.float32)

buf = vd.asbuffer(arrs)
buf2 = vd.asbuffer(arrs2)

def add_shader(self: vd.shader_builder, buf, buf2):
    val: vd.shader_variable = buf[self.global_x]
    val2 = buf2[self.global_x]

    self.printf(f"Val {self.global_x.format}: {val.format}", self.global_x, val)

    buf2[self.global_x] += buf[self.global_x]

vd.dispatch_shader(add_shader, [1, 1, 1], [16, 1, 1], [buf, buf2])

print("Diff: ", np.mean(np.abs(arrs + arrs2 - buf2.read(0))))




