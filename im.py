import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

#from vkdispatch.codegen import f32, v4
#from vkdispatch.codegen import Buffer as Buff

import numpy as np

from matplotlib import pyplot as plt

arr: np.ndarray = np.load("data/bronwyn/template_3d.npy") # np.random.rand(512, 512, 512).astype(np.float32)

#vd.initialize(log_level=vd.LogLevel.INFO)

arr_buff = vd.Buffer(arr.shape, vd.float32)
image = vd.Image3D(arr.shape, vd.float32, 1)

@vc.shader(exec_size=lambda args: args.buff.size)
def my_shader(buff: Buff[f32], img: Img3[f32], sigma: Const[f32], mag: Var[f32]):
    ind = vc.global_invocation.x.copy()
    sample_coords = vc.unravel_index(ind, buff.shape).cast_to(v4)
    buff[ind] = img.sample(sample_coords).x * sigma + mag

print(my_shader)

image.write(arr)

my_shader(arr_buff, image, 1.0, 0.0)

plt.imshow(arr[0, :, :])
plt.show()

ret = arr_buff.read()[0]

plt.imshow(ret[0, :, :])
plt.show()

print(ret.shape)

print(np.sum(np.abs(arr - ret)))
