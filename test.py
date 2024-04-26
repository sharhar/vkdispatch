import vkdispatch as vd 
import numpy as np

arrs = np.random.rand(4, 4).astype(np.float32)
arrs2 = np.random.rand(4, 4).astype(np.float32)

buf = vd.asbuffer(arrs)
buf2 = vd.asbuffer(arrs2)

@vd.compute_shader(buf, buf2, local_size=(16, 1, 1))
def add_shader(self: vd.shader_builder, buf, buf2):
    self.printf(f"Val {self.global_x.format}: {buf[self.global_x].format}", self.global_x, buf[self.global_x])
    buf2[self.global_x] += buf[self.global_x] * buf2[self.global_x] / 7

add_shader[buf.size]()

print("Diff: ", np.mean(np.abs(arrs * arrs2 / 7 + arrs2 - buf2.read(0))))




