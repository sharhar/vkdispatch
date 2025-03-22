

import numpy as np

N = 16

signal = np.ones((N,), dtype=np.complex64)
signal[N//16] = 0
reference_data = np.fft.fft(signal).reshape(-1, 16)

a = np.array([
    16.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, 15.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j, -1.+0.j
])

even = a[:16]
odd = a[16:]

factor = np.exp(-2j * np.pi * np.arange(32) / 32)

print(even)
print(odd)

first_half = even + factor[:16] * odd
second_half = even + factor[16:] * odd

#print(a)

print(first_half)
print(second_half)
#print(reference_data)
