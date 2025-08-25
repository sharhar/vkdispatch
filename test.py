import vkdispatch as vd
import tqdm

buff = vd.Buffer((1024, 1024), var_type=vd.complex64)

#vd.fft.fft(buff, axis=0, print_shader=True)
vd.vkfft.fft(buff, axis=0, print_shader=True)