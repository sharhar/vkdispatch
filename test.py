import vkdispatch as vd 
import numpy as np

arrs = np.random.rand(4, 4).astype(np.float32)
arrs2 = np.random.rand(4, 4).astype(np.float32)
arrs3 = np.random.rand(4, 4).astype(np.float32)
arrs4 = np.random.rand(4, 4).astype(np.float32)

buf = vd.asbuffer(arrs)
buf2 = vd.asbuffer(arrs2)
buf3 = vd.asbuffer(arrs3)
buf4 = vd.asbuffer(arrs4)


@vd.compute_shader(vd.float32[0], vd.float32[0])
def add_buffers(buf, buf2):
    ind = vd.shader.global_x.copy()
    num = vd.shader.push_constant(vd.float32, "num")
    #vd.shader.print(ind, " ", num)
    buf2[ind] = buf[ind] * buf2[ind] / num

cmd_list = vd.command_list()

add_buffers[buf.size, cmd_list](buf, buf2, num=34.0)
add_buffers[buf.size, cmd_list](buf2, buf4, num=7.0)

cmd_list.submit()

print("Diff: ", np.mean(np.abs(arrs * arrs2 / 34.0 - buf2.read(0))))
print("Diff2: ", np.mean(np.abs(buf2.read(0) * arrs4 / 7.0 - buf4.read(0))))




