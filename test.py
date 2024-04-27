import vkdispatch as vd 
import numpy as np

arrs = np.random.rand(4, 4).astype(np.float32)
arrs2 = np.random.rand(4, 4).astype(np.float32)

buf = vd.asbuffer(arrs)
buf2 = vd.asbuffer(arrs2)


#@vd.compute_shader(vd.float32[], vd.float32[])
@vd.compute_shader(buf, buf2)
def add_buffers(self: vd.shader_builder, buf, buf2):
    #self.print("Val ", self.global_x, ": ", buf[self.global_x])
    ind = self.global_x
    
    #test = self._make_var(vd.int32)
    #test2 = test

    #test2 += 1

    #self.print("Test: ", test)
    #self.print("Test2: ", test2)

    num = self.dynamic_constant(vd.float32, "num")

    self.print("Num: ", num)

    buf2[ind] += buf[ind] * buf2[ind] / num

cmd_list = vd.command_list()

pc = add_buffers[buf.size, cmd_list]()

pc["num"] = 34

cmd_list.submit()

print("Diff: ", np.mean(np.abs(arrs * arrs2 / 34 + arrs2 - buf2.read(0))))




