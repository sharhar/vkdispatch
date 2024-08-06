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

def test_dims(dims: tuple):
    ref_arr = np.ones(shape=dims, dtype=np.complex64) #make_random_complex_signal(dims)

    test_signal = vd.Buffer(ref_arr.shape, vd.complex64)
    test_signal.write(ref_arr)

    # Perform an FFT on the buffer
    vd.fft(test_signal)
    vd.ifft(test_signal)

    return np.abs(test_signal.read(0) - ref_arr).mean() + 1

data_points = []

#for dim_size in tqdm.tqdm(range(250, 259)):
for dim_size in range(100, 120) :
    print(dim_size, test_dims((dim_size, dim_size)))
    #data_points.append(test_dims((dim_size, dim_size)))

#np.save("data.npy", np.array(data_points))

plt.plot(np.log(data_points))
plt.show()