import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

from matplotlib import pyplot as plt

vd.initialize(debug_mode=True)

N = 64
B = 32
signal = np.zeros((B, N,), dtype=np.complex64)

signal[:, N//4:N//3] = 1

test_data = vd.asbuffer(signal)

@vd.map_registers([vd.complex64])
def my_map(buffer: vc.Buffer[vc.c64]):
    vc.mapping_registers()[0][:] = buffer[vc.mapping_index()] * 2

#vd.fft.fft(test_data, test_data, print_shader=True, input_map=my_map)
vd.fft.fft(test_data, print_shader=True)

data = test_data.read(0)
reference_data = np.fft.fft(signal)

diff_arr = np.abs(data - reference_data)

plt.imshow(np.abs(reference_data))
plt.colorbar()
plt.show()

plt.imshow(np.abs(data))
plt.colorbar()
plt.show()

print(np.allclose(data, reference_data, atol=1e-2))
