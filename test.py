import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np
import tqdm

from matplotlib import pyplot as plt

import sys

def make_random_complex_signal(shape):
    r = np.random.random(size=shape)
    i = np.random.random(size=shape)
    return (r + i * 1j).astype(np.complex64)

def make_signal_circle(shape):
    r = np.zeros(shape, dtype=np.complex64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i - shape[0] // 2) ** 2 + (j - shape[1] // 2) ** 2 < 100:
                r[i, j] = 1 + 0j
    return r

def test_dims(dims: tuple):
    ref_arr = make_signal_circle(dims)  #make_random_complex_signal(dims)

    test_signal = vd.Buffer(ref_arr.shape, vd.complex64)
    test_signal.write(ref_arr)

    # Perform an FFT on the buffer
    vd.fft(test_signal)
    vd.ifft(test_signal)

    return np.abs(test_signal.read(0) / (np.prod(ref_arr.shape)) - ref_arr).mean() + 1

data_points = []

for dim_size in tqdm.tqdm(range(384, 389)):
#for dim_size in range(100, 120) :
    #print(dim_size, test_dims((dim_size, dim_size)))
    data_points.append(test_dims((dim_size, dim_size)))

#np.save("data.npy", np.array(data_points))

plt.plot(np.log(data_points))
plt.show()