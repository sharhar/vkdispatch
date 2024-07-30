import vkdispatch as vd
import numpy as np

@vd.compute_shader(vd.complex64[0])
def fill_buffer(buf):
    buf[vd.shader.global_x] = vd.shader.static_constant(vd.complex64, "val")

@vd.compute_shader(vd.int32[0], vd.float32[0])
def place_atoms(image, atom_coords):
    ind = vd.shader.global_x.copy()

    rotation_matrix = vd.shader.push_constant(vd.mat4, "rot_matrix")
    pix_size = vd.shader.push_constant(vd.float32, "pix_size")

    pos = vd.shader.new(vd.vec4)
    pos.x = -atom_coords[3*ind + 1] / pix_size
    pos.y = atom_coords[3*ind + 0] / pix_size
    pos.z = atom_coords[3*ind + 2] / pix_size
    pos.w = 1

    pos[:] = rotation_matrix * pos

    image_ind = vd.shader.new(vd.ivec2)
    image_ind.y = vd.shader.ceil(pos.y).cast_to(vd.int32) + (image.shape.x / 2)
    image_ind.x = vd.shader.ceil(-pos.x).cast_to(vd.int32) + (image.shape.y / 2)

    vd.shader.if_any(image_ind.x < 0, image_ind.x >= image.shape.x, image_ind.y < 0, image_ind.y >= image.shape.y)
    vd.shader.return_statement()
    vd.shader.end()

    vd.shader.atomic_add(image[2 * image_ind.x, 2 * image_ind.y], 1)

@vd.compute_shader(vd.float32[0])
def convert_int_to_float(image):
    ind = vd.shader.global_x.copy()
    image[ind] = vd.shader.float_bits_to_int(image[ind]).cast_to(vd.float32)

@vd.compute_shader(vd.complex64[0])
def apply_gaussian_filter(buf: vd.ShaderVariable):
    ind = vd.shader.global_x.cast_to(vd.int32).copy()

    var = vd.shader.static_constant(vd.float32, "var")

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

    my_dist = vd.shader.new(vd.float32)
    my_dist[:] = (x_norm*x_norm + y_norm*y_norm) / (2 * var)

    vd.shader.if_statement(my_dist > 100)
    buf[ind].real = 0
    buf[ind].imag = 0
    vd.shader.return_statement()
    vd.shader.end()

    buf[ind] *= vd.shader.exp(-my_dist) / (buf.shape.x * buf.shape.y)

class TemplatePotential:
    def __init__(self, atom_coords: np.ndarray, template_shape, B, pix_size, A=3909) -> None:
        self.template_shape = template_shape
        self.pix_size = pix_size # [A]
        self.A = A # amplitude for potential scaling

        self.B = B + 8 * (np.pi * 0.27)**2 + (self.pix_size)**2 / 12 # [A^2] blurring B-factor with contributions from pixel size and physical PSF

        self.exec_size = template_shape[0] * template_shape[1]

        self.atom_coords = atom_coords
        self.atom_coords_buffer = vd.asbuffer(atom_coords)
        self.rotation_matrix = None

        self.var = 2 * pix_size**2 / self.B # [dimensionless] gaussian blur variance
    
    def record(self, cmd_list: vd.CommandList, work_buffer: vd.Buffer, rot_matrix: np.ndarray = None):
        fill_buffer[self.exec_size, cmd_list](work_buffer, val=0)

        if rot_matrix is not None:
            place_atoms[self.atom_coords.shape[0], cmd_list](work_buffer, self.atom_coords_buffer, pix_size=self.pix_size, rot_matrix=rot_matrix)
        else:
            self.rotation_matrix = place_atoms[self.atom_coords.shape[0], cmd_list](work_buffer, self.atom_coords_buffer, pix_size=self.pix_size)

        convert_int_to_float[work_buffer.size * 2, cmd_list](work_buffer)

        vd.fft[cmd_list](work_buffer)
        apply_gaussian_filter[work_buffer.size, cmd_list](work_buffer, var=self.var)
        vd.ifft[cmd_list](work_buffer)

    def set_rotation_matrix(self, rot_matrix: np.ndarray):
        self.rotation_matrix["rot_matrix"] = rot_matrix