import vkdispatch
import vkdispatch_native
import numpy as np

device_num = len(vkdispatch.get_devices())

ctx = vkdispatch.context([i for i in range(device_num)], [1 for i in range(device_num)])

buf = vkdispatch.buffer(ctx, (512, 512, 512), np.float32)
buf2 = vkdispatch.buffer(ctx, (512, 512, 512), np.float32)

arrs = [np.random.rand(512, 512, 512).astype(np.float32) for _ in range(device_num)]

for i in range(device_num):
    buf.write(arrs[i], i)

comp_func = lambda ind1, ind2: print(f"{ind1} - {ind2}: {np.mean(np.abs(arrs[ind1] - buf.read(ind2)))}")

for i in range(device_num):
    for j in range(device_num):
        comp_func(i, j)


comp_func = lambda ind1, ind2: print(f"{ind1} - {ind2}: {np.mean(np.abs(arrs[ind1] - buf2.read(ind2)))}")

for i in range(device_num):
    for j in range(device_num):
        comp_func(i, j)

buf.copy_to(buf2, 0)

comp_func = lambda ind1, ind2: print(f"{ind1} - {ind2}: {np.mean(np.abs(arrs[ind1] - buf2.read(ind2)))}")

for i in range(device_num):
    for j in range(device_num):
        comp_func(i, j)