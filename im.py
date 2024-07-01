import vkdispatch as vd

import numpy as np

from matplotlib import pyplot as plt

arr = np.load("data/bronwyn/template_3d.npy") # np.random.rand(512, 512, 512).astype(np.float32)

#vd.initialize(log_level=vd.LogLevel.INFO)

print(arr.shape)

arr_buff = vd.Buffer(arr.shape, vd.float32)

image = vd.Image3D(arr.shape, vd.float32, 1) #(vd.image_format.R32_SFLOAT, (512, 512), vd.image_tiling.LINEAR, vd.image_usage.SAMPLED | vd.image_usage.TRANSFER_DST, vd.memory_property.DEVICE_LOCAL)

@vd.compute_shader(vd.float32[0])
def test_shader(buff):
    ind = vd.shader.global_x.copy()

    img = vd.shader.texture_sampler(3)

    x = (ind % 576).copy()
    y = ((ind / 576) % 576).copy()
    z = (ind / (576 * 576)).copy()

    sample_coords = vd.shader.new(vd.vec4, x, y, z, 0)

    #vd.shader.print(sample_coords)

    buff[ind] = img.sample(sample_coords).x

    #vd.shader.print(buff[ind])

image.write(arr)

test_shader[arr_buff.size](arr_buff, image_bindings=[(2, image)])

plt.imshow(arr[0, :, :])
plt.show()


ret = arr_buff.read()[0]

#ret = image.read()

plt.imshow(ret[0, :, :])
plt.show()

print(ret.shape)

print(np.sum(np.abs(arr - ret)))
