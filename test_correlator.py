import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

@vc.shader(exec_size=lambda args: args.max_cross.size)
def init_accumulators(max_cross: Buff[f32], best_index: Buff[i32]):
    ind = vc.global_invocation.x.copy()
    max_cross[ind] = -1000000
    best_index[ind] = -1

@vc.shader(exec_size=lambda args: args.input.size)
def cross_correlate(input: Buff[c64], reference: Buff[c64]):
    ind = vc.global_invocation.x.copy()

    input_val = vc.new(vd.complex64)
    input_val[:] = input[ind] / (input.shape.x * input.shape.y)

    input[ind].real = input_val.real * reference[ind].real + input_val.imag * reference[ind].imag
    input[ind].imag = input_val.imag * reference[ind].real - input_val.real * reference[ind].imag

@vc.shader(exec_size=lambda args: args.output.size)
def fftshift_and_crop(output: Buff[c64], input: Buff[c64]):
    ind = vc.global_invocation.x.cast_to(vd.int32).copy()

    out_x = (ind / output.shape.y).copy()
    out_y = (ind % output.shape.y).copy()

    vc.if_any(vc.logical_and(
                out_x >= input.shape.x / 2,
                out_x < output.shape.x - input.shape.x / 2),
             vc.logical_and(
                out_y >= input.shape.y / 2,
                out_y < output.shape.y - input.shape.y / 2))
    output[ind].real = 0.0
    output[ind].imag = 0.0
    vc.return_statement()
    vc.end()

    in_x = ((out_x + input.shape.x / 2) % output.shape.x).copy()
    in_y = ((out_y + input.shape.y / 2) % output.shape.y).copy()

    output[ind] = input[in_x, in_y]

@vc.shader(exec_size=lambda args: args.back_buffer.size)
def update_max(max_cross: Buff[f32], best_index: Buff[i32], back_buffer: Buff[c64], index: Var[i32]):
    ind = vc.global_invocation.x.copy()

    current_cross_correlation = back_buffer[ind].real

    vc.if_statement(current_cross_correlation > max_cross[ind])
    max_cross[ind] = current_cross_correlation
    best_index[ind] = index
    vc.end()
        

class Correlator:
    def __init__(self, reference_image: np.ndarray) -> None:

        self.match_image_buffer = vd.asbuffer(reference_image.astype(np.complex64))
        vd.fft(self.match_image_buffer)

        self.shift_buffer = vd.Buffer(reference_image.shape, vd.complex64)
        self.max_cross = vd.Buffer(reference_image.shape, vd.float32)
        self.best_index = vd.Buffer(reference_image.shape, vd.int32)

        init_accumulators(self.max_cross, self.best_index)
    
    def record(self, work_buffer: vd.Buffer, index: int):
        fftshift_and_crop(self.shift_buffer, work_buffer)

        vd.fft(self.shift_buffer)
        cross_correlate(self.shift_buffer, self.match_image_buffer)
        vd.ifft(self.shift_buffer)

        update_max(self.max_cross, self.best_index, self.shift_buffer, index=index)
