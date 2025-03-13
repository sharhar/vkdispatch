import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

row_size = 1024 * 1024 * 8

#data = np.array([
#    list(range(0, row_size)),
#    list(range(row_size, row_size * 2)),
#    list(range(row_size * 2, row_size * 3)),
#], dtype=np.float32)

np.random.seed(0)

data = np.random.rand(3, row_size).astype(np.float32)

print(data)

data_buffer = vd.asbuffer(data)

axes = (1,)
@vd.reduce(0, axes=axes)
def reduce_sum(a: f32, b: f32) -> f32:
    return (a + b).copy()

reduction_result = reduce_sum(data_buffer)
# Expected output:
print(np.sum(data, axis=axes).flatten() - reduction_result.read(0)[0])

print(reduction_result.read(0)[0])
print(np.sum(data, axis=axes))