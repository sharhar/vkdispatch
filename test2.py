import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

from matplotlib import pyplot as plt

vd.initialize(debug_mode=True)

N = 242
B = 13 * 5
signal = np.zeros((5, 242, 13), dtype=np.complex64)

signal[:, N//4:N//3, :] = 1

test_data = vd.asbuffer(signal)

#vd.fft.fft(test_data, test_data, print_shader=True, input_map=my_map)
vd.fft.fft(test_data, axis=1, print_shader=True)

data = test_data.read(0)
reference_data = np.fft.fft(signal, axis=1)

diff_arr = np.abs(data - reference_data)

plt.imshow(np.abs(reference_data[0]))
plt.colorbar()
plt.show()

plt.imshow(np.abs(data[0] - reference_data[0]))
plt.colorbar()
plt.show()

print(np.allclose(data, reference_data, atol=1e-2))
