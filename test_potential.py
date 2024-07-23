import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

import typing

@vc.shader(exec_size=lambda args: args.buf.size)
def fill_buffer(buf: Buff[c64], val: Const[c64] = 0):
    buf[vc.global_invocation.x] = val

@vc.shader(exec_size=lambda args: args.atom_coords.shape[0])
def place_atoms(image: Buff[i32], atom_coords: Buff[f32], rot_matrix: Var[m4]):
    ind = vc.global_invocation.x.copy()

    pos = vc.new_vec4() #shader.new(vd.vec4)
    pos.x = -atom_coords[3*ind + 1] 
    pos.y = atom_coords[3*ind + 0]
    pos.z = atom_coords[3*ind + 2]
    pos.w = 1

    pos[:] = rot_matrix * pos

    image_ind = vc.new_ivec2()
    image_ind.y = vc.ceil(pos.y).cast_to(vd.int32) + (image.shape.x / 2)
    image_ind.x = vc.ceil(-pos.x).cast_to(vd.int32) + (image.shape.y / 2)

    vc.if_any(image_ind.x < 0, image_ind.x >= image.shape.x, image_ind.y < 0, image_ind.y >= image.shape.y)
    vc.return_statement()
    vc.end()

    vc.atomic_add(image[2 * image_ind.x, 2 * image_ind.y], 1)

@vc.shader(exec_size=lambda args: args.work_buffer.size * 2)
def convert_int_to_float(work_buffer: Buff[f32]):
    ind = vc.global_invocation.x.copy()
    work_buffer[ind] = vc.float_bits_to_int(work_buffer[ind]).cast_to(vd.float32)

#@vd.compute_shader(vd.complex64[0])
@vc.shader(exec_size=lambda args: args.buf.size)
def apply_gaussian_filter(buf: Buff[c64], sigma: Const[f32], mag: Const[f32]):
    ind = vc.global_invocation.x.copy()

    x = (ind / buf.shape.y).copy()
    y = (ind % buf.shape.y).copy()

    x[:] = x + buf.shape.x / 2
    y[:] = y + buf.shape.y / 2

    x[:] = x % buf.shape.x
    y[:] = y % buf.shape.y

    x[:] = x - buf.shape.x / 2
    y[:] = y - buf.shape.y / 2

    x_norm = (x.cast_to(vd.float32) / buf.shape.x.cast_to(vd.float32)).copy()
    y_norm = (y.cast_to(vd.float32) / buf.shape.y.cast_to(vd.float32)).copy()

    my_dist = vc.new_float()
    my_dist[:] = (x_norm*x_norm + y_norm*y_norm) / ( sigma * sigma / 2 )

    vc.if_statement(my_dist > 100)
    buf[ind].real = 0
    buf[ind].imag = 0
    vc.return_statement()
    vc.end()

    buf[ind] *= mag * vc.exp(-my_dist) / (buf.shape.x * buf.shape.y)

class TemplatePotential:
    def __init__(self, atom_coords: np.ndarray, mag, sigma) -> None:
        self.mag = mag
        self.sigma = sigma

        self.atom_coords = atom_coords
        self.atom_coords_buffer = vd.asbuffer(atom_coords)
    
    def record(self,work_buffer: vd.Buffer, rot_matrix: np.ndarray):
        fill_buffer(work_buffer)
        place_atoms(work_buffer, self.atom_coords_buffer, rot_matrix=rot_matrix)
        convert_int_to_float(work_buffer)
        
        vd.fft(work_buffer)
        apply_gaussian_filter(work_buffer, sigma=self.sigma, mag=self.mag)
        vd.ifft(work_buffer)
