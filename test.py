import vkdispatch as vd
import vkdispatch.codegen as vc

vd.initialize(backend="vulkan", log_level=vd.LogLevel.INFO)
vc.set_codegen_backend("glsl")

SIZE = 4096

buff_shape = (2, SIZE, SIZE)

buff = vd.Buffer(buff_shape, var_type=vd.complex64)

vd.vkfft.fft(buff, axis=1) #, print_shader=True)

vd.queue_wait_idle()

#print(vd.fft.fft_src(buff_shape, axis=1).code)