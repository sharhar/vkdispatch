import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

from matplotlib import pyplot as plt

#arr = np.load("data/bronwyn/template_3d.npy") # np.random.rand(512, 512, 512).astype(np.float32)

#vd.initialize(log_level=vd.LogLevel.INFO)

#print(arr.shape)

#arr_buff = vd.Buffer(arr.shape, vd.float32)

#image = vd.Image3D(arr.shape, vd.float32, 1)

#@vd.compute_shader(vd.float32[0]

@vc.shader(local_size=(1, 1, 1))
def my_shader(buff: vc.Buffer[vd.float32],
              img: vc.Image3D[vd.float32],
              sigma: vc.Constant[vd.mat4],
              mag: vc.Variable[vd.float32]):

    ind = vc.global_invocation.x

    buff[ind] = img.sample(vc.global_invocation)



print(my_shader)



#@vd.shader
#def test_shader(buff: vd.Buffer[vd.float32],
#                #img: vd.Image3D[vd.float32],
#                sigma: vd.Constant[vd.float32],
#                mag: vd.Variable[vd.float32]):
#    pass

"""
    ind = vd.shader.global_x.copy()
    #img = vd.shader.texture_sampler(3)

    x = (ind % 576).copy()
    y = ((ind / 576) % 576).copy()
    z = (ind / (576 * 576)).copy()

    sample_coords = vd.shader.new(vd.vec4, x, y, z, 0)

    tt = buff[ind]

    print(tt)

    tester = img.sample(sample_coords).x * sigma + mag

    print(tester)
"""


#test_shader(None, 1.0, 1.0)

#image.write(arr)

#test_shader[arr_buff.size](arr_buff, image_bindings=[(2, image)])

#plt.imshow(arr[0, :, :])
#plt.show()


#ret = arr_buff.read()[0]

#ret = image.read()

#plt.imshow(ret[0, :, :])
#plt.show()

#print(ret.shape)

#print(np.sum(np.abs(arr - ret)))
