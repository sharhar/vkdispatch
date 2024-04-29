import vkdispatch as vd 
from matplotlib import pyplot as plt
import numpy as np
import sys
import tf_calc

file_out = sys.argv[1]
input_image_raw = np.load(sys.argv[2])

n_std = 5
outliers_idxs = np.abs(input_image_raw - input_image_raw.mean()) > n_std * input_image_raw.std()
input_image_raw[outliers_idxs] = input_image_raw.mean()
test_image_normalized: np.ndarray = (input_image_raw - input_image_raw.mean()) / input_image_raw.std()

phi_values = np.arange(0, 360, 2.5)
theta_values = np.arange(0, 180, 2.5)
psi_values = np.arange(0, 360, 1.5)

atom_data = np.load(sys.argv[3])
atom_coords = atom_data["coords"].astype(np.float32)
atom_proton_counts = atom_data["proton_counts"].astype(np.float32)

atom_coords -= np.sum(atom_coords, axis=0) / atom_coords.shape[0]

tf_data = tf_calc.prepareTF(input_image_raw.shape, 1.056, 0, upsample_factor=1)

tf_data_array = np.zeros(shape=(tf_data[0].shape[0], tf_data[0].shape[1], 9), dtype=np.float32) #vd.asbuffer(tf_data)
tf_data_array[:, :, 0] = tf_data[0]
tf_data_array[:, :, 1] = tf_data[1]
tf_data_array[:, :, 2] = tf_data[2]
tf_data_array[:, :, 3] = tf_data[3]
tf_data_array[:, :, 4] = tf_data[4]
tf_data_array[:, :, 5] = tf_data[5]
tf_data_array[:, :, 6] = tf_data[6]
tf_data_array[:, :, 7] = tf_data[7]
tf_data_array[:, :, 8] = tf_data[8]

atom_coords_buffer = vd.asbuffer(atom_coords)
tf_data_buffer = vd.asbuffer(tf_data_array)

work_buffer = vd.buffer(tf_data[0].shape, vd.complex64)


def get_rotation_matrix(angles: list[int], offsets: list[int] = [0, 0]):
    in_matricies = np.zeros(shape=(16,), dtype=np.float32)

    cos_phi   = np.cos(np.deg2rad(angles[0]))
    sin_phi   = np.sin(np.deg2rad(angles[0]))
    cos_theta = np.cos(np.deg2rad(angles[1]))
    sin_theta = np.sin(np.deg2rad(angles[1]))

    M00 = cos_phi * cos_theta 
    M01 = -sin_phi 

    M10 = sin_phi * cos_theta 
    M11 = cos_phi 

    M20 = -sin_theta 

    cos_psi_in_plane   = np.cos(np.deg2rad(-angles[2] - 90)) 
    sin_psi_in_plane   = np.sin(np.deg2rad(-angles[2] - 90))

    m00  = cos_psi_in_plane
    m01 = sin_psi_in_plane
    m10 = -sin_psi_in_plane
    m11 = cos_psi_in_plane

    in_matricies[0] = m00 * M00 + m10 * M01
    in_matricies[1] = m01 * M00 + m11 * M01
    
    in_matricies[4] = m00 * M10 + m10 * M11
    in_matricies[5] = m01 * M10 + m11 * M11
    
    in_matricies[8] = m00 * M20 
    in_matricies[9] = m01 * M20

    in_matricies[12] = offsets[0]
    in_matricies[13] = offsets[1]

    return in_matricies




cmd_list = vd.command_list()

@vd.compute_shader(vd.complex64[0])
def fill_buffer(buf):
    buf[vd.shader.global_x] = vd.shader.push_constant(vd.complex64, "val")

fill_buffer[work_buffer.size, cmd_list](work_buffer, val=0)

@vd.compute_shader(vd.int32[0], vd.float32[0])
def place_atoms(image, atom_coords):
    ind = vd.shader.global_x.copy()

    rotation_matrix = vd.shader.push_constant(vd.mat4, "rot_matrix")

    pos = vd.shader.new(vd.vec4)
    pos[0] = atom_coords[3*ind + 0]
    pos[1] = atom_coords[3*ind + 1]
    pos[2] = atom_coords[3*ind + 2]
    pos[3] = 1

    pos[:] = rotation_matrix * pos

    image_ind = vd.shader.new(vd.ivec2)
    image_ind[0] = vd.shader.ceil(pos[0]).cast_to(vd.int32) + (work_buffer.shape[0] // 2)
    image_ind[1] = vd.shader.ceil(pos[1]).cast_to(vd.int32) + (work_buffer.shape[1] // 2)

    vd.shader.if_statement(image_ind[0] < 0 or image_ind[0] >= work_buffer.shape[0] or image_ind[1] < 0 or image_ind[1] >= work_buffer.shape[1])
    vd.shader.return_statement()
    vd.shader.end_if()

    vd.shader.atomic_add(image[2 * image_ind[1] * work_buffer.shape[0] + 2 * image_ind[0]], 1)

place_atoms[atom_coords.shape[0], cmd_list](work_buffer, atom_coords_buffer, rot_matrix=get_rotation_matrix([0, 0, 0], [0, 0]))

cmd_list.submit()


plt.imshow(np.abs(work_buffer.read(0)))
plt.show()




"""
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

@vd.compute_shader(vd.float32[0])
def inplace_mult(buf):
    num = vd.shader.push_constant(vd.float32, "num")
    buf[vd.shader.global_x.copy()] *=  num

cmd_list = vd.command_list()

add_buffers[buf.size, cmd_list](buf, buf2, num=34.0)
inplace_mult[buf.size, cmd_list](buf2, num=5.0)
add_buffers[buf.size, cmd_list](buf2, buf4, num=7.0)

cmd_list.submit()

print("Diff: ", np.mean(np.abs(5 * arrs * arrs2 / 34.0 - buf2.read(0))))
print("Diff2: ", np.mean(np.abs(buf2.read(0) * arrs4 / 7.0 - buf4.read(0))))

"""





