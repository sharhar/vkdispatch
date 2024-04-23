import vkdispatch as vd 
import numpy as np

device_num = len(vd.get_devices())

buf = vd.buffer((512, 512), np.float32)
buf2 = vd.buffer((512, 512), np.float32)

#fft_plan = vd.fft_plan((512, 512))

cmdList = vd.command_list()

arrs = np.random.rand(512, 512).astype(np.float32)
arrs2 = np.random.rand(512, 512).astype(np.float32)

shader_source = r"""
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) buffer Buffer1 {
    float data[];
} buf1;

layout(set = 0, binding = 1) buffer Buffer2 {
    float data[];
} buf2;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint index = gl_GlobalInvocationID.x;
    buf2.data[index] += buf1.data[index];
}

"""

compute_plan = vd.compute_plan(shader_source, 2, 0)

compute_plan.bind_buffer(buf, 0)
compute_plan.bind_buffer(buf2, 1)

#arrs_fft = [np.fft.fft2(arr) for arr in arrs]

buf.write(arrs, 0)
buf2.write(arrs2, 0)

print("Arrs: ", np.mean(np.abs(arrs)))
print("Arrs2: ", np.mean(np.abs(arrs2)))

print("Buf: ", np.mean(np.abs(buf.read(0))))
print("Buf2: ", np.mean(np.abs(buf2.read(0))))

print("Comp: ", np.mean(np.abs(arrs - buf.read(0))))
print("Comp2: ", np.mean(np.abs(arrs2 - buf2.read(0))))

compute_plan.record(cmdList, (512 * 2, 1, 1))

cmdList.submit()

print("Add: ", np.mean(np.abs(arrs + arrs2 - buf2.read(0))))
