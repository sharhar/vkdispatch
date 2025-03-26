import numpy as np
import vkdispatch as vd

from matplotlib import pyplot as plt

offset_count = []

for i in range(1, 10001):
    offset_count.append(vd.fft.pad_dim(i) - i)

offsets_arr = np.array(offset_count).reshape((100, 100))

offsets_bins = offsets_arr.mean(axis=1)

#plt.imshow(offsets_arr)

#plt.plot(offsets_bins)
plt.bar(range(100), offsets_bins)
plt.title('Average padding added to FFT dimension (from 1 to 10001, window size = 100)')
plt.xlabel('FFT dimension')
plt.ylabel('Padding added')
plt.show()
