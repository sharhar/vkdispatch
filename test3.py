import warp as wp
import numpy as np

num_points = 1024

# Define Warp kernel to compute the length of 4D points
@wp.kernel
def length_kernel(points: wp.array(dtype=wp.vec4),
           lengths: wp.array(dtype=float)):
    tid = wp.tid()
    lengths[tid] = wp.length(points[tid])

# generate random points
points = np.random.rand(num_points, 4).astype(np.float32)

# allocate arrays in Warp
points_array = wp.array(points, dtype=wp.vec4)
lengths_array = wp.zeros(num_points, dtype=float)

# launch warp kernel
wp.launch(kernel=length_kernel,
          dim=num_points, # specify execution size
          inputs=[points_array, lengths_array])

print(lengths_array)


import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

num_points = 1024

# Define a vkdispatch shader to compute the length of 4D points
@vd.shader("points.size")
def length_shader(points: vc.Buff[vc.v4], lengths: vc.Buff[vc.f32]):
    tid = vc.global_invocation().x
    lengths[tid] = vc.length(points[tid])

# generate random points
# points = np.random.rand(num_points, 4).astype(np.float32)

# allocate buffers in vkdispatch
points_buffer = vd.Buffer((num_points, ), var_type=vd.vec4)
points_buffer.write(points)
lengths_buffer = vd.asbuffer(np.zeros(num_points, dtype=np.float32))

# launch vkdispatch shader
length_shader(
    points_buffer,
    lengths_buffer,
)

print(lengths_buffer.read(0))