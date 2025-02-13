import vkdispatch as vd

vd.make_context(use_cpu=True)

import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

#print("Running test_basic")

vd.initialize(log_level=vd.LogLevel.INFO, debug_mode=True)

#print("Creating command stream")

# Create a 1D image
#signal = np.sin(np.array([i/8 for i in range(0, 50, 1)])).astype(np.float32)

#test_line = vd.Image1D(len(signal), vd.float32)
#test_line.write(signal)

#assert np.allclose(test_line.read(0), signal)

# Create a 2D image
signal_2d = np.sin(np.array([[i/8 + j/17 for i in range(0, 50, 1)] for j in range(0, 50, 1)])).astype(np.float32)
sample_factor = 10

test_img = vd.Image2D(signal_2d.shape, vd.float32)
test_img.write(signal_2d)

result_arr = vd.Buffer((signal_2d.shape[0] * (sample_factor - 1), signal_2d.shape[1] * (sample_factor - 1)), vd.float32)

@vd.shader(exec_size=lambda args: args.buff.size)
def do_approx(buff: Buff[f32], img: Img2[f32]):
    ind = vc.global_invocation().x.copy()
    ind_2d = vc.unravel_index(ind, buff.shape)
    buff[ind] = img.sample((ind_2d.cast_to(v2)) / sample_factor).x

do_approx(result_arr, test_img.sample())

signal_full = np.sin(np.array([[i/80 + j/170 for i in range(0, 450, 1)] for j in range(0, 450, 1)])).astype(np.float32)

assert np.allclose(result_arr.read()[0], signal_full, atol=0.0025)
