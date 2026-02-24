import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

vd.initialize(backend="dummy")

vd.set_dummy_context_params(max_workgroup_size=(64, 1, 1))

fft_srcs = [
    vd.fft.fft_src((2 ** i,))
    for i in range(4, 12)
]

print("FFT shader sources:", fft_srcs)