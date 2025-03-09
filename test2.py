
#data = [0.9127319, 0.7329998, 0.0011958987, 0.2081132, 0.56774485, 0.070007905, 0.18449862, 0.19299245, 0.26252803, 0.46872616, 0.03796457, 0.16031821, 0.5665843, 0.46797255, 0.9101622, 0.21621224, 0.034223225, 0.2636695, 0.42257962, 0.8933027, 0.83871573, 0.5227679, 0.8838187, 0.11580994, 0.9424848, 0.6193833, 0.51532704, 0.2923926, 0.17214341, 0.4877235, 0.3948569, 0.9867599, 0.5824371, 0.78653204, 0.2081846, 0.253086, 0.27474797, 0.16537598, 0.7246974, 0.7372597, 0.12605102, 0.73903614, 0.7202814, 0.05744897, 0.4580918, 0.028030781, 0.30317643, 0.2792883, 0.27650988, 0.5226246]

data = list(range(50))

print(sum(data)) # + sum(data[-2:]))

import vkdispatch as vd

vd.initialize(debug_mode=True)

import vkdispatch.codegen as vc
import numpy as np

buff = vd.asbuffer(np.array(data, dtype=np.float32))

@vd.reduce(0, group_size=1024 + 512)
def sum_reduce(a: vc.f32, b: vc.f32) -> vc.f32:
    return a + b

print(sum_reduce)

import time
time.sleep(1)

res_buf = sum_reduce(buff)

time.sleep(0.5)

print(res_buf.read(0), sum(data))

time.sleep(0.5)


exit()

import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

from matplotlib import pyplot as plt

vd.initialize(debug_mode=True)

side_length = 624

signal_shape = (side_length, side_length)

x = np.linspace(-1, 1, signal_shape[0])
y = np.linspace(-1, 1, signal_shape[1])
x, y = np.meshgrid(x, y)
d = np.sqrt(x*x + y*y)
sigma, mu = 0.02, 0.0
gaussian_signal = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))

mid_idx = side_length // 2

square_signal = np.zeros(signal_shape)
square_signal[mid_idx - 64:mid_idx + 64, mid_idx - 64:mid_idx + 64] = 1

dot_signal = np.zeros(signal_shape, dtype=np.float32)
dot_signal[mid_idx, mid_idx] = 1

input_signal = np.fft.fftshift(dot_signal).astype(np.float32)
kernel_signal = np.fft.fftshift(square_signal).astype(np.float32)

convolved_signal = np.fft.ifftshift(np.fft.ifft2(
    (np.fft.fft2(input_signal).astype(np.complex64) 
    * np.fft.fft2(kernel_signal).astype(np.complex64))
    .astype(np.complex64)))

input_buffer = vd.asrfftbuffer(input_signal)
kernel_buffer = vd.asrfftbuffer(kernel_signal.reshape(1, side_length, -1))
output_buffer = vd.asrfftbuffer(np.ones(signal_shape).astype(np.float32))

vd.prepare_convolution_kernel(kernel_buffer)
vd.convolve_2d(input_buffer, kernel_buffer) #, input_buffer)

result = np.fft.ifftshift(input_buffer.read_real(0))

plt.imshow(result / side_length - np.abs(convolved_signal))
plt.show()

exit()