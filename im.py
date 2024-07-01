import vkdispatch as vd

import numpy as np

arr = np.random.rand(512, 512).astype(np.float32)

vd.initialize(log_level=vd.LogLevel.INFO)

image = vd.Image2D((512, 512), np.float32, 1) #(vd.image_format.R32_SFLOAT, (512, 512), vd.image_tiling.LINEAR, vd.image_usage.SAMPLED | vd.image_usage.TRANSFER_DST, vd.memory_property.DEVICE_LOCAL)

image.write(arr)

ret = image.read()

print(np.sum(np.abs(arr - ret)))