import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

from matplotlib import pyplot as plt

signal = np.sin(np.array([i/4 for i in range(0, 50, 2)])).astype(np.float32)
sample_factor = 10
padding_size = 50

test_line = vd.Image1D(len(signal), vd.float32)
test_line.write(signal)

result_arr = vd.Buffer((len(signal) * sample_factor + 2 * padding_size,), vd.float32)

@vc.shader(exec_size=lambda args: args.buff.size)
def do_approx(buff: Buff[f32], line: Img1[f32]):
    ind = vc.global_invocation.x.copy()
    buff[ind] = line.sample((ind.cast_to(f32) - padding_size) / sample_factor).x

do_approx(result_arr, test_line)

plt.scatter(np.array([i for i in range(result_arr.shape[0])]), result_arr.read()[0])
plt.scatter(np.array([i for i in range(padding_size, len(signal) * sample_factor + padding_size, sample_factor)]), signal)
plt.show()
