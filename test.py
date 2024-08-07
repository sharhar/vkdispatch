import vkdispatch as vd 
import numpy as np

device_num = len(vd.get_devices())

buff_shape = (256, 256)

buf = vd.buffer(buff_shape, np.complex64)

fft_plan = vd.fft_plan(buff_shape)

cmdList = vd.command_list()

def make_2d_circle_signal(shape, radius):
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    xx, yy = np.meshgrid(x, y)
    return (xx ** 2 + yy ** 2) < radius ** 2

from matplotlib import pyplot as plt

arrs = [make_2d_circle_signal(buff_shape, 0.5).astype(np.complex64) for _ in range(device_num)]
arrs_fft = [np.fft.fft2(arr) for arr in arrs]

for i in range(device_num):
    buf.write(arrs[i], i)

comp_func = lambda ind1, ind2: print(f"{ind1} - {ind2}: {np.mean(np.abs(arrs_fft[ind1] - buf.read(ind2)))}")

for i in range(device_num):
    for j in range(device_num):
        comp_func(i, j)


#plt.imshow(np.abs(arrs[0]))
#plt.show()

fft_plan.record_forward(cmdList, buf)

plt.imshow(np.log(np.abs(buf.read(0)) + 1))
plt.show()

#cmdList.submit()

#buf.copy_to(buf2, 0)

comp_func = lambda ind1, ind2: print(f"{ind1} - {ind2}: {np.mean(np.abs(arrs_fft[ind1] - buf.read(ind2)))}")
