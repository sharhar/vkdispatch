import sys
import time
import typing

import numpy as np
import tqdm
from matplotlib import pyplot as plt

import tf_calc
import vkdispatch as vd

current_time = time.time()

sigma_e = tf_calc.get_sigmaE(300e3)
amp_ratio = -0.07

file_out = sys.argv[1]
input_image_raw: np.ndarray = np.load(sys.argv[2])

n_std = 5
outliers_idxs = (
    np.abs(input_image_raw - input_image_raw.mean()) > n_std * input_image_raw.std()
)
input_image_raw[outliers_idxs] = input_image_raw.mean()
test_image_normalized: np.ndarray = (
    input_image_raw - input_image_raw.mean()
) / input_image_raw.std()

test_image_normalized = np.fft.fftshift(test_image_normalized)

mag = 200
sigma = 0.025

# phi_values = np.arange(0, 360, 2.5)
# theta_values = np.arange(0, 180, 2.5)
# psi_values = np.arange(0, 360, 1.5)

phi_values = np.arange(0, 100, 5)
theta_values = np.arange(0, 90, 5)
psi_values = np.arange(0, 100, 5)

defocus_values = np.arange(11000, 13000, 400)
test_values = np.array(
    np.meshgrid(phi_values, theta_values, psi_values, defocus_values)
).T.reshape(-1, 4)

atom_data = np.load(sys.argv[3])
atom_coords: np.ndarray = atom_data["coords"].astype(np.float32)
atom_proton_counts: np.ndarray = atom_data["proton_counts"].astype(np.float32)

atom_coords -= np.sum(atom_coords, axis=0) / atom_coords.shape[0]

tf_data: typing.Tuple[np.ndarray] = tf_calc.prepareTF(input_image_raw.shape, 1.056, 0)

tf_data_array = np.zeros(
    shape=(tf_data[0].shape[0], tf_data[0].shape[1], 9), dtype=np.float32
)  # vd.asbuffer(tf_data)
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
match_image_buffer = vd.asbuffer(test_image_normalized.astype(np.complex64))

vd.fft(match_image_buffer)

work_buffer = vd.Buffer(input_image_raw.shape, vd.complex64)
shift_buffer = vd.Buffer(input_image_raw.shape, vd.complex64)
max_cross = vd.Buffer(input_image_raw.shape, vd.float32)
best_index = vd.Buffer(input_image_raw.shape, vd.int32)

workgroup_size = vd.get_devices()[0].max_workgroup_size[0]
assert work_buffer.size % workgroup_size == 0

reduce_buffer = vd.Buffer(((work_buffer.size // workgroup_size) + 1,), vd.vec2)


@vd.compute_shader(vd.float32[0], vd.int32[0])
def init_accumulators(max_cross, best_index):
    ind = vd.shader.global_x.copy()
    max_cross[ind] = -1000000
    best_index[ind] = -1


init_accumulators[max_cross.size](max_cross, best_index)


def get_rotation_matrix(angles: typing.List[int], offsets: typing.List[int] = [0, 0]):
    in_matricies = np.zeros(shape=(4, 4), dtype=np.float32)

    cos_phi = np.cos(np.deg2rad(angles[0]))
    sin_phi = np.sin(np.deg2rad(angles[0]))
    cos_theta = np.cos(np.deg2rad(angles[1]))
    sin_theta = np.sin(np.deg2rad(angles[1]))

    M00 = cos_phi * cos_theta
    M01 = -sin_phi

    M10 = sin_phi * cos_theta
    M11 = cos_phi

    M20 = -sin_theta

    cos_psi_in_plane = np.cos(np.deg2rad(-angles[2] - 90))
    sin_psi_in_plane = np.sin(np.deg2rad(-angles[2] - 90))

    m00 = cos_psi_in_plane
    m01 = sin_psi_in_plane
    m10 = -sin_psi_in_plane
    m11 = cos_psi_in_plane

    in_matricies[0, 0] = m00 * M00 + m10 * M01
    in_matricies[0, 1] = m00 * M10 + m10 * M11
    in_matricies[0, 2] = m00 * M20
    in_matricies[0, 3] = offsets[0]

    in_matricies[1, 0] = m01 * M00 + m11 * M01
    in_matricies[1, 1] = m01 * M10 + m11 * M11
    in_matricies[1, 2] = m01 * M20
    in_matricies[1, 3] = offsets[1]

    return in_matricies.T


@vd.compute_shader(vd.complex64[0])
def fill_buffer(buf):
    buf[vd.shader.global_x] = vd.shader.push_constant(vd.complex64, "val")


@vd.compute_shader(vd.float32[0], vd.float32[0])
def place_atoms(image, atom_coords):
    ind = vd.shader.global_x.copy()

    rotation_matrix = vd.shader.push_constant(vd.mat4, "rot_matrix")

    pos = vd.shader.new(vd.vec4)
    pos[0] = -atom_coords[3 * ind + 1]
    pos[1] = atom_coords[3 * ind + 0]
    pos[2] = atom_coords[3 * ind + 2]
    pos[3] = 1

    pos[:] = rotation_matrix * pos

    image_ind = vd.shader.new(vd.ivec2)
    image_ind[0] = vd.shader.ceil(pos[1]).cast_to(vd.int32) + (
        work_buffer.shape[0] // 2
    )
    image_ind[1] = vd.shader.ceil(-pos[0]).cast_to(vd.int32) + (
        work_buffer.shape[1] // 2
    )

    vd.shader.if_any(
        image_ind[0] < 0,
        image_ind[0] >= work_buffer.shape[0],
        image_ind[1] < 0,
        image_ind[1] >= work_buffer.shape[1],
    )
    vd.shader.return_statement()
    vd.shader.end_if()

    vd.shader.atomic_add(
        image[2 * image_ind[1] * work_buffer.shape[0] + 2 * image_ind[0]], 1
    )


@vd.compute_shader(vd.complex64[0])
def apply_gaussian_filter(buf):
    ind = vd.shader.global_x.cast_to(vd.int32).copy()

    x = (ind % work_buffer.shape[0]).copy()
    y = (ind / work_buffer.shape[1]).copy()

    x[:] = x + work_buffer.shape[0] // 2
    y[:] = y + work_buffer.shape[1] // 2

    x[:] = x % work_buffer.shape[0]
    y[:] = y % work_buffer.shape[1]

    x[:] = x - work_buffer.shape[0] // 2
    y[:] = y - work_buffer.shape[1] // 2

    my_dist = vd.shader.new(vd.float32)
    my_dist[:] = (x * x + y * y) * sigma * sigma / 2

    vd.shader.if_statement(my_dist > 100)
    buf[ind][0] = 0
    buf[ind][1] = 0
    vd.shader.return_statement()
    vd.shader.end_if()

    buf[ind] *= (
        mag * vd.shader.exp(-my_dist) / (work_buffer.shape[0] * work_buffer.shape[1])
    )


@vd.compute_shader(vd.complex64[0])
def potential_to_wave(image):
    ind = vd.shader.global_x.copy()

    potential = (image[ind][0] * sigma_e).copy()

    A = vd.shader.exp(amp_ratio * potential).copy()

    image[ind][0] = A * vd.shader.cos(potential)
    image[ind][1] = A * vd.shader.sin(potential)


@vd.compute_shader(vd.complex64[0])
def mult_by_mask(image):
    ind = vd.shader.global_x.copy()

    r = (ind / work_buffer.shape[1]).copy()
    c = (ind % work_buffer.shape[1]).copy()

    vd.shader.if_statement(r > work_buffer.shape[0] // 2)
    r -= work_buffer.shape[0]
    vd.shader.end_if()

    vd.shader.if_statement(c > work_buffer.shape[1] // 2)
    c -= work_buffer.shape[1]
    vd.shader.end_if()

    rad_sq = (r * r + c * c).copy()

    vd.shader.if_statement(rad_sq > (work_buffer.shape[0] * work_buffer.shape[1] // 16))
    image[ind][0] = 0
    image[ind][1] = 0
    vd.shader.end_if()


@vd.compute_shader(vd.complex64[0], vd.float32[0])
def apply_transfer_function(image, tf_data):
    ind = vd.shader.global_x.copy()

    defocus = vd.shader.push_constant(vd.float32, "defocus")

    V1_r_scaler = tf_data[9 * ind + 0]
    V1_c_scaler = tf_data[9 * ind + 1]

    V1_r_adder = tf_data[9 * ind + 2]
    V1_c_adder = tf_data[9 * ind + 3]

    mag_pre = tf_data[9 * ind + 4]
    V_scaler = tf_data[9 * ind + 5]

    gamma_pre_scaler = tf_data[9 * ind + 6]
    gamma_pre_adder = tf_data[9 * ind + 7]

    eta_tot = tf_data[9 * ind + 8]

    V1_r = V1_r_scaler * defocus + V1_r_adder
    V1_c = V1_c_scaler * defocus + V1_c_adder

    mag = (mag_pre * vd.shader.exp(V_scaler * (V1_r * V1_r + V1_c * V1_c))).copy()
    mag /= work_buffer.shape[0] * work_buffer.shape[1]
    gamma = gamma_pre_scaler * defocus + gamma_pre_adder

    phase = (-gamma - eta_tot).copy()

    rot_vec = vd.shader.new(vd.vec2)
    rot_vec[0] = vd.shader.cos(phase)
    rot_vec[1] = vd.shader.sin(phase)

    wv = vd.shader.new(vd.vec2)
    wv[:] = image[ind]
    image[ind][0] = mag * (wv[0] * rot_vec[0] - wv[1] * rot_vec[1])
    image[ind][1] = mag * (wv[0] * rot_vec[1] + wv[1] * rot_vec[0])


@vd.compute_shader(vd.vec2[0], vd.complex64[0])
def normalization_stage1(reduce_buffer, image):
    ind = vd.shader.global_x.copy()

    sdata = vd.shader.shared_buffer(
        vd.vec2, workgroup_size // vd.get_devices()[0].sub_group_size
    )

    sum_vec = vd.shader.new(vd.vec2)
    sum_vec[0] = (
        image[ind][0] * image[ind][0] + image[ind][1] * image[ind][1]
    )  # / (work_buffer.shape[0] * work_buffer.shape[1])
    sum_vec[1] = sum_vec[0] * sum_vec[0]

    image[ind][0] = sum_vec[0]
    image[ind][1] = 0

    sum_vec[:] = vd.shader.subgroup_add(sum_vec)

    vd.shader.if_statement(vd.shader.subgroup_invocation == 0)
    sdata[vd.shader.subgroup_id] = sum_vec
    vd.shader.end_if()

    vd.shader.memory_barrier_shared()
    vd.shader.barrier()

    vd.shader.if_statement(vd.shader.subgroup_id == 0)
    sum_vec[:] = "vec2(0.0, 0.0)"

    vd.shader.if_statement(vd.shader.subgroup_invocation < vd.shader.num_subgroups)
    sum_vec[:] = sdata[vd.shader.subgroup_invocation]
    vd.shader.end_if()

    sum_vec[:] = vd.shader.subgroup_add(sum_vec)
    vd.shader.end_if()

    vd.shader.if_statement(vd.shader.local_x == 0)
    reduce_buffer[vd.shader.workgroup_x] = sum_vec
    vd.shader.end_if()


@vd.compute_shader(vd.vec2[0], local_size=(reduce_buffer.shape[0] - 1, 1, 1))
def normalization_stage2(buff):
    ind = vd.shader.global_x.copy()

    sdata = vd.shader.shared_buffer(
        vd.vec2, workgroup_size // vd.get_devices()[0].sub_group_size
    )  # vd.shader.num_subgroups)

    sum_vec = buff[ind].copy()

    sum_vec[:] = vd.shader.subgroup_add(sum_vec)

    vd.shader.if_statement(vd.shader.subgroup_invocation == 0)
    sdata[vd.shader.subgroup_id] = sum_vec
    vd.shader.end_if()

    vd.shader.memory_barrier_shared()
    vd.shader.barrier()

    vd.shader.if_statement(vd.shader.subgroup_id == 0)
    sum_vec[:] = "vec2(0.0, 0.0)"

    vd.shader.if_all(vd.shader.subgroup_invocation < vd.shader.num_subgroups)
    sum_vec[:] = sdata[vd.shader.subgroup_invocation]
    vd.shader.end_if()

    sum_vec[:] = vd.shader.subgroup_add(sum_vec)
    vd.shader.end_if()

    vd.shader.memory_barrier_shared()
    vd.shader.barrier()

    vd.shader.if_statement(vd.shader.local_x == 0)
    buff[reduce_buffer.shape[0] - 1] = sum_vec
    vd.shader.end_if()


@vd.compute_shader(vd.complex64[0], vd.vec2[0])
def normalization_stage3(image, buff):
    ind = vd.shader.global_x.copy()

    sum_vec = (buff[reduce_buffer.shape[0] - 1] / work_buffer.size).copy()
    sum_vec[1] = vd.shader.sqrt(sum_vec[1] - sum_vec[0] * sum_vec[0])

    image[ind][0] = (image[ind][0] - sum_vec[0]) / sum_vec[1]


@vd.compute_shader(vd.complex64[0], vd.complex64[0])
def cross_correlate(input, reference):
    ind = vd.shader.global_x.copy()

    input_val = vd.shader.new(vd.complex64)
    input_val[:] = input[ind] / work_buffer.size

    input[ind][0] = input_val[0] * reference[ind][0] + input_val[1] * reference[ind][1]
    input[ind][1] = input_val[1] * reference[ind][0] - input_val[0] * reference[ind][1]


@vd.compute_shader(vd.complex64[0], vd.complex64[0])
def fftshift(output, input):
    ind = vd.shader.global_x.cast_to(vd.int32).copy()

    r = (ind / work_buffer.shape[1]).copy()
    c = (ind % work_buffer.shape[1]).copy()

    r[:] = (r + work_buffer.shape[0] // 2) % work_buffer.shape[0]
    c[:] = (c + work_buffer.shape[1] // 2) % work_buffer.shape[1]

    output[ind] = input[r * work_buffer.shape[1] + c]


@vd.compute_shader(vd.float32[0], vd.int32[0], vd.complex64[0])
def update_max(max_cross, best_index, back_buffer):
    ind = vd.shader.global_x.copy()

    current_cross_correlation = back_buffer[ind][0]

    vd.shader.if_statement(current_cross_correlation > max_cross[ind])
    max_cross[ind] = current_cross_correlation
    best_index[ind] = vd.shader.push_constant(vd.int32, "index")
    vd.shader.end_if()


cmd_list = vd.CommandList()

fill_buffer[work_buffer.size, cmd_list](work_buffer, val=0)
rotation_matrix = place_atoms[atom_coords.shape[0], cmd_list](
    work_buffer, atom_coords_buffer
)

vd.fft[cmd_list](work_buffer)
apply_gaussian_filter[work_buffer.size, cmd_list](work_buffer)
vd.ifft[cmd_list](work_buffer)

potential_to_wave[work_buffer.size, cmd_list](work_buffer)

vd.fft[cmd_list](work_buffer)
mult_by_mask[work_buffer.size, cmd_list](work_buffer)
defocus = apply_transfer_function[work_buffer.size, cmd_list](
    work_buffer, tf_data_buffer
)
vd.ifft[cmd_list](work_buffer)

normalization_stage1[work_buffer.size, cmd_list](reduce_buffer, work_buffer)
normalization_stage2[reduce_buffer.size - 1, cmd_list](reduce_buffer)
normalization_stage3[work_buffer.size, cmd_list](work_buffer, reduce_buffer)

fftshift[work_buffer.size, cmd_list](shift_buffer, work_buffer)

vd.fft[cmd_list](shift_buffer)
cross_correlate[shift_buffer.size, cmd_list](shift_buffer, match_image_buffer)
vd.ifft[cmd_list](shift_buffer)

fftshift[shift_buffer.size, cmd_list](work_buffer, shift_buffer)

template_index = update_max[work_buffer.size, cmd_list](
    max_cross, best_index, work_buffer
)

batch_size = 100

status_bar = tqdm.tqdm(total=test_values.shape[0])

for i in range(0, test_values.shape[0], batch_size):
    data = b""

    for j in range(batch_size):
        if i + j >= test_values.shape[0]:
            break

        rotation_matrix["rot_matrix"] = get_rotation_matrix(
            test_values[i + j][:3], [0, 0]
        )
        defocus["defocus"] = test_values[i + j][3]
        template_index["index"] = i + j

        for pc_buffer in cmd_list.pc_buffers:
            data += pc_buffer.get_bytes()

    cmd_list.submit(data=data)

    status_bar.update(batch_size)

status_bar.close()

final_max_cross = max_cross.read(0)
best_index_result = best_index.read(0)
index_of_max = np.unravel_index(np.argmax(final_max_cross), final_max_cross.shape)
final_index = best_index_result[index_of_max]

print("Found max at:", index_of_max)
print("Max cross correlation:", final_max_cross[index_of_max])
print("Final index:", final_index)
print("Phi:", test_values[final_index][0])
print("Theta:", test_values[final_index][1])
print("Psi:", test_values[final_index][2])
print("Defocus:", test_values[final_index][3])

fill_buffer[work_buffer.size](work_buffer, val=0)
place_atoms[atom_coords.shape[0]](
    work_buffer, atom_coords_buffer, rot_matrix=get_rotation_matrix([118, 95, 294])
)  # test_values[final_index][:3]))

vd.fft(work_buffer)
apply_gaussian_filter[work_buffer.size](work_buffer)
vd.ifft(work_buffer)

potential_to_wave[work_buffer.size](work_buffer)

vd.fft(work_buffer)
mult_by_mask[work_buffer.size](work_buffer)
apply_transfer_function[work_buffer.size](
    work_buffer, tf_data_buffer, defocus=test_values[final_index][3]
)
vd.ifft(work_buffer)

normalization_stage1[work_buffer.size](reduce_buffer, work_buffer)
normalization_stage2[reduce_buffer.size - 1](reduce_buffer)
normalization_stage3[work_buffer.size](work_buffer, reduce_buffer)

params_result = test_values[best_index_result]

np.save(file_out + "_mip.npy", final_max_cross)
np.save(file_out + "_match.npy", work_buffer.read(0))
np.save(file_out + "_best_index.npy", best_index_result)
np.save(file_out + "_phi.npy", params_result[:, :, 0])
np.save(file_out + "_theta.npy", params_result[:, :, 1])
np.save(file_out + "_psi.npy", params_result[:, :, 2])
np.save(file_out + "_defocus.npy", params_result[:, :, 3])

plt.imshow(final_max_cross)
plt.colorbar()
plt.show()

# (4084, 5749)
# 3000000 template
# 16 hours