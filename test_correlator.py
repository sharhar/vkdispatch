import vkdispatch as vd
import numpy as np

class Correlator:
    def __init__(self, template_shape, reference_image: np.ndarray) -> None:  
        @vd.compute_shader(vd.float32[0], vd.int32[0])
        def init_accumulators(max_cross, best_index):
            ind = vd.shader.global_x.copy()
            max_cross[ind] = -1000000
            best_index[ind] = -1

        @vd.compute_shader(vd.complex64[0], vd.complex64[0])
        def cross_correlate(input, reference):
            ind = vd.shader.global_x.copy()

            input_val = vd.shader.new(vd.complex64)
            input_val[:] = input[ind] / (template_shape[0] * template_shape[1])

            input[ind][0] = input_val[0] * reference[ind][0] + input_val[1] * reference[ind][1]
            input[ind][1] = input_val[1] * reference[ind][0] - input_val[0] * reference[ind][1]

        @vd.compute_shader(vd.complex64[0], vd.complex64[0])
        def fftshift_and_crop(output, input):
            ind = vd.shader.global_x.cast_to(vd.int32).copy()

            out_x = (ind / reference_image.shape[1]).copy()
            out_y = (ind % reference_image.shape[1]).copy()

            vd.shader.if_any(vd.shader.logical_and(
                                out_x >= template_shape[0] // 2,
                                out_x < reference_image.shape[0] - template_shape[0] // 2),
                            vd.shader.logical_and(
                                out_y >= template_shape[1] // 2,
                                out_y < reference_image.shape[1] - template_shape[1] // 2))
            output[ind][0] = 0.0
            output[ind][1] = 0.0
            vd.shader.return_statement()
            vd.shader.end()

            in_x = ((out_x + template_shape[0] // 2) % reference_image.shape[0]).copy()
            in_y = ((out_y + template_shape[1] // 2) % reference_image.shape[1]).copy()

            output[ind] = input[in_x * template_shape[1] + in_y]

        @vd.compute_shader(vd.float32[0], vd.int32[0], vd.complex64[0])
        def update_max(max_cross, best_index, back_buffer):
            ind = vd.shader.global_x.copy()

            current_cross_correlation = back_buffer[ind][0]

            vd.shader.if_statement(current_cross_correlation > max_cross[ind])
            max_cross[ind] = current_cross_correlation
            best_index[ind] = vd.shader.push_constant(vd.int32, "index")
            vd.shader.end()
        
        self.init_accumulators = init_accumulators
        self.cross_correlate = cross_correlate
        self.fftshift_and_crop = fftshift_and_crop
        self.update_max = update_max

        self.match_image_buffer = vd.asbuffer(reference_image.astype(np.complex64))
        vd.fft(self.match_image_buffer)

        self.shift_buffer = vd.Buffer(reference_image.shape, vd.complex64)
        self.max_cross = vd.Buffer(reference_image.shape, vd.float32)
        self.best_index = vd.Buffer(reference_image.shape, vd.int32)

        init_accumulators[self.max_cross.size](self.max_cross, self.best_index)

        self.template_index = None
    
    def record(self, cmd_list: vd.CommandList, work_buffer: vd.Buffer, index: int = None):
        self.fftshift_and_crop[self.shift_buffer.size, cmd_list](self.shift_buffer, work_buffer)

        vd.fft[cmd_list](self.shift_buffer)
        self.cross_correlate[self.shift_buffer.size, cmd_list](self.shift_buffer, self.match_image_buffer)
        vd.ifft[cmd_list](self.shift_buffer)

        if index is not None:
            self.update_max[self.shift_buffer.size, cmd_list](self.max_cross, self.best_index, self.shift_buffer, index=index)
        else:
            self.template_index = self.update_max[self.shift_buffer.size, cmd_list](self.max_cross, self.best_index, self.shift_buffer)

    def set_index(self, index: int):
        self.template_index["index"] = index