import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

import typing

@vc.shader(exec_size=lambda args: args.image.size)
def potential_to_wave(image: Buff[c64], sigma_e: Const[f32], amp_ratio: Const[f32]):
    ind = vc.global_invocation.x.copy()

    potential = (image[ind].real * sigma_e).copy()

    A = vc.exp(amp_ratio * potential).copy()

    image[ind].real = A * vc.cos(potential)
    image[ind].imag = A * vc.sin(potential)

@vc.shader(exec_size=lambda args: args.image.size)
def mult_by_mask(image: Buff[c64]):
    ind = vc.global_invocation.x.copy()

    r = (ind / image.shape.y).copy()
    c = (ind % image.shape.y).copy()

    vc.if_statement(r > image.shape.x / 2)
    r -= image.shape.x
    vc.end()

    vc.if_statement(c > image.shape.y / 2)
    c -= image.shape.y
    vc.end()

    rad_sq = (r*r + c*c).copy()

    vc.if_statement(rad_sq > (image.shape.x * image.shape.y / 16))
    image[ind].real = 0
    image[ind].imag = 0
    vc.end()

@vc.shader(exec_size=lambda args: args.image.size)
def apply_transfer_function(image: Buff[c64], tf_data: Buff[f32], defocus: Var[f32]):
    ind = vc.global_invocation.x.copy()

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

    mag = (mag_pre * vc.exp(V_scaler * (V1_r * V1_r + V1_c * V1_c))).copy()
    mag /= image.shape.x * image.shape.y
    gamma = gamma_pre_scaler * defocus + gamma_pre_adder

    phase = (-gamma - eta_tot).copy()

    rot_vec = vc.new(vd.vec2)
    rot_vec.x = vc.cos(phase)
    rot_vec.y = vc.sin(phase)

    wv = vc.new(vd.vec2)
    wv[:] = image[ind]
    image[ind].real = mag * (wv.x * rot_vec.x - wv.y * rot_vec.y)
    image[ind].imag = mag * (wv.x * rot_vec.y + wv.y * rot_vec.x)

@vc.map_reduce(exec_size=lambda args: args.wave.size, reduction="subgroupAdd") # We define the reduction function here
def calc_sums(ind: Const[i32], wave: Buff[v2]) -> v2: # so this is the mapping function
    result = vc.new_vec2()
    result.x = wave[ind].x * wave[ind].x + wave[ind].y * wave[ind].y
    result.y = result.x * result.x

    wave[ind].x = result.x
    wave[ind].y = 0

    return result

@vc.shader(exec_size=lambda args: args.image.size)
def normalize_image(image: Buff[v2], sum_buff: Buff[v2]):
    ind = vc.global_invocation.x.copy()

    sum_vec = (sum_buff[0] / (image.shape.x * image.shape.y)).copy()
    sum_vec.y = vc.sqrt(sum_vec.y - sum_vec.x * sum_vec.x)

    image[ind].x = (image[ind].x - sum_vec.x) / sum_vec.y

class Scope:
    def __init__(self, tf_data: typing.Tuple[np.ndarray], sigma_e: float, amp_ratio: float) -> None:
        self.sigma_e = sigma_e
        self.amp_ratio = amp_ratio
        
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

    def record(self, work_buffer: vd.Buffer, defocus: float):
        potential_to_wave(work_buffer, sigma_e=self.sigma_e, amp_ratio=self.amp_ratio)

        vd.fft(work_buffer)
        mult_by_mask(work_buffer)
        apply_transfer_function(work_buffer, self.tf_data_buffer, defocus=defocus)
        vd.ifft(work_buffer)

        self.sum_buff = calc_sums(work_buffer) # The reduction returns a buffer with the result in the first value
        normalize_image(work_buffer, self.sum_buff)
    
    

