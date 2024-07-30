import vkdispatch as vd
import numpy as np

import typing

@vd.compute_shader(vd.complex64[0])
def potential_to_wave(image):
    ind = vd.shader.global_x.copy()

    sigma_e = vd.shader.push_constant(vd.float32, "sigma_e")
    amp_ratio = vd.shader.push_constant(vd.float32, "amp_ratio")

    potential = (image[ind].real * sigma_e).copy()

    A = vd.shader.exp(-amp_ratio * potential).copy()

    image[ind].real = A * vd.shader.cos(potential)
    image[ind].imag = A * vd.shader.sin(potential)

@vd.compute_shader(vd.complex64[0])
def mult_by_mask(image):
    ind = vd.shader.global_x.copy()

    r = (ind / image.shape.y).copy()
    c = (ind % image.shape.y).copy()

    vd.shader.if_statement(r > image.shape.x / 2)
    r -= image.shape.x
    vd.shader.end()

    vd.shader.if_statement(c > image.shape.y / 2)
    c -= image.shape.y
    vd.shader.end()

    rad_sq = (r*r + c*c).copy()

    vd.shader.if_statement(rad_sq > (image.shape.x * image.shape.y / 16))
    image[ind].real = 0
    image[ind].imag = 0
    vd.shader.end()

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
    mag /= image.shape.x * image.shape.y
    gamma = gamma_pre_scaler * defocus + gamma_pre_adder

    phase = (-gamma - eta_tot).copy()

    rot_vec = vd.shader.new(vd.vec2)
    rot_vec.x = vd.shader.cos(phase)
    rot_vec.y = vd.shader.sin(phase)

    wv = vd.shader.new(vd.vec2)
    wv[:] = image[ind]
    image[ind].real = mag * (wv.x * rot_vec.x - wv.y * rot_vec.y)
    image[ind].imag = mag * (wv.x * rot_vec.y + wv.y * rot_vec.x)

@vd.map_reduce(vd.vec2, vd.vec2[0], reduction="subgroupAdd") # We define the reduction function here
def calc_sums(wave): # so this is the mapping function
    result = vd.shader.new(vd.vec2)
    result.x = wave.x * wave.x + wave.y * wave.y
    result.y = result.x * result.x

    wave.x = result.x
    wave.y = 0

    return result

@vd.compute_shader(vd.vec2[0], vd.vec2[0])
def normalize_image(image, sum_buff):
    ind = vd.shader.global_x.copy()

    sum_vec = (sum_buff[0] / (image.shape.x * image.shape.y)).copy()
    sum_vec.y = vd.shader.sqrt(sum_vec.y - sum_vec.x * sum_vec.x)

    image[ind].x = (image[ind].x - sum_vec.x) / sum_vec.y


class Scope:
    def __init__(self, template_shape, tf_data: typing.Tuple[np.ndarray], sigma_e: float, amp_ratio: float) -> None:
        self.template_shape = template_shape
        self.sigma_e = sigma_e
        self.amp_ratio = amp_ratio

        self.exec_size = template_shape[0] * template_shape[1]
        
        self.defocus = None
        self.sum_buff = None

        tf_data_array = np.zeros(shape=(tf_data[0].shape[0], tf_data[0].shape[1], 9), dtype=np.float32)
        tf_data_array[:, :, 0] = tf_data[0]
        tf_data_array[:, :, 1] = tf_data[1]
        tf_data_array[:, :, 2] = tf_data[2]
        tf_data_array[:, :, 3] = tf_data[3]
        tf_data_array[:, :, 4] = tf_data[4]
        tf_data_array[:, :, 5] = tf_data[5]
        tf_data_array[:, :, 6] = tf_data[6]
        tf_data_array[:, :, 7] = tf_data[7]
        tf_data_array[:, :, 8] = tf_data[8]

        self.tf_data_buffer = vd.asbuffer(tf_data_array)

    def record(self, cmd_list: vd.CommandList, work_buffer: vd.Buffer, defocus: float = None):
        potential_to_wave[self.exec_size, cmd_list](work_buffer, sigma_e=self.sigma_e, amp_ratio=self.amp_ratio)

        vd.fft[cmd_list](work_buffer)
        mult_by_mask[work_buffer.size, cmd_list](work_buffer)
        
        if defocus is not None:
            apply_transfer_function[work_buffer.size, cmd_list](work_buffer, self.tf_data_buffer, defocus=defocus)
        else:
            self.defocus = apply_transfer_function[work_buffer.size, cmd_list](work_buffer, self.tf_data_buffer)
        
        vd.ifft[cmd_list](work_buffer)

        self.sum_buff = calc_sums[work_buffer.size, cmd_list](work_buffer) # The reduction returns a buffer with the result in the first value
        normalize_image[work_buffer.size, cmd_list](work_buffer, self.sum_buff)
    
    def set_defocus(self, defocus: float):
        self.defocus["defocus"] = defocus
    

