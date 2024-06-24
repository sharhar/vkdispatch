import vkdispatch as vd
import numpy as np

class TemplatePotential:
    def __init__(self, atom_coords: np.ndarray, template_shape, mag, sigma) -> None:
        @vd.compute_shader(vd.complex64[0])
        def fill_buffer(buf):
            buf[vd.shader.global_x] = vd.shader.push_constant(vd.complex64, "val")

        @vd.compute_shader(vd.int32[0], vd.float32[0])
        def place_atoms(image, atom_coords):
            ind = vd.shader.global_x.copy()

            rotation_matrix = vd.shader.push_constant(vd.mat4, "rot_matrix")

            pos = vd.shader.new(vd.vec4)
            pos[0] = -atom_coords[3*ind + 1] 
            pos[1] = atom_coords[3*ind + 0]
            pos[2] = atom_coords[3*ind + 2]
            pos[3] = 1

            pos[:] = rotation_matrix * pos

            image_ind = vd.shader.new(vd.ivec2)
            image_ind[1] = vd.shader.ceil(pos[1]).cast_to(vd.int32) + (template_shape[0] // 2)
            image_ind[0] = vd.shader.ceil(-pos[0]).cast_to(vd.int32) + (template_shape[1] // 2)

            vd.shader.if_any(image_ind[0] < 0, image_ind[0] >= template_shape[0], image_ind[1] < 0, image_ind[1] >= template_shape[1])
            vd.shader.return_statement()
            vd.shader.end()

            vd.shader.atomic_add(image[2 * (image_ind[0] * template_shape[1] + image_ind[1])], 1)

        @vd.compute_shader(vd.float32[0])
        def convert_int_to_float(image):
            ind = vd.shader.global_x.copy()
            image[ind] = vd.shader.float_bits_to_int(image[ind]).cast_to(vd.float32)
        
        @vd.compute_shader(vd.complex64[0])
        def apply_gaussian_filter(buf):
            ind = vd.shader.global_x.cast_to(vd.int32).copy()

            x = (ind / template_shape[1]).copy()
            y = (ind % template_shape[1]).copy()

            x[:] = x + template_shape[0] // 2
            y[:] = y + template_shape[1] // 2

            x[:] = x % template_shape[0]
            y[:] = y % template_shape[1]

            x[:] = x - template_shape[0] // 2
            y[:] = y - template_shape[1] // 2

            x_norm = (x.cast_to(vd.float32) / float(template_shape[0])).copy()
            y_norm = (y.cast_to(vd.float32) / float(template_shape[1])).copy()

            my_dist = vd.shader.new(vd.float32)
            my_dist[:] = (x_norm*x_norm + y_norm*y_norm) / ( sigma * sigma / 2 )

            vd.shader.if_statement(my_dist > 100)
            buf[ind][0] = 0
            buf[ind][1] = 0
            vd.shader.return_statement()
            vd.shader.end()

            buf[ind] *= mag * vd.shader.exp(-my_dist) / (template_shape[0] * template_shape[1])

        self.fill_buffer = fill_buffer
        self.place_atoms = place_atoms
        self.convert_int_to_float = convert_int_to_float
        self.apply_gaussian_filter = apply_gaussian_filter

        self.exec_size = template_shape[0] * template_shape[1]

        self.atom_coords = atom_coords
        self.atom_coords_buffer = vd.asbuffer(atom_coords)
        self.rotation_matrix = None
    
    def record(self, cmd_list: vd.CommandList, work_buffer: vd.Buffer, rot_matrix: np.ndarray = None):
        self.fill_buffer[self.exec_size, cmd_list](work_buffer, val=0)

        if rot_matrix is not None:
            self.place_atoms[self.atom_coords.shape[0], cmd_list](work_buffer, self.atom_coords_buffer, rot_matrix=rot_matrix)
        else:
            self.rotation_matrix = self.place_atoms[self.atom_coords.shape[0], cmd_list](work_buffer, self.atom_coords_buffer)

        self.convert_int_to_float[work_buffer.size * 2, cmd_list](work_buffer)

        vd.fft[cmd_list](work_buffer)
        self.apply_gaussian_filter[work_buffer.size, cmd_list](work_buffer)
        vd.ifft[cmd_list](work_buffer)

    def set_rotation_matrix(self, rot_matrix: np.ndarray):
        self.rotation_matrix["rot_matrix"] = rot_matrix
