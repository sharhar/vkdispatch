import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

#vd.initialize(log_level=vd.LogLevel.VERBOSE, debug_mode=True)

vd.log_info("Buffer1")
buf = vd.Buffer((1024,) , v2)
vd.log_info("Buffer2")
buf2 = vd.Buffer((1,) , v2)

# Create a numpy array
data = np.random.rand(1024, 2).astype(np.float32)

# Write the data to the buffer
buf.write(data)

@vd.map_reduce(
        exec_size=lambda args: args.buffer.size, 
        reduction=lambda x, y: x + y, 
        reduction_identity=0
)
def sum_map(ind: Const[i32], buffer: Buff[v2]) -> v2:
    return vc.sin(buffer[ind])



cmd_stream = vd.CommandStream()

res_buf = sum_map(buf, cmd_stream=cmd_stream)

cmd_stream.submit()

read_data = res_buf.read(0)

# Check that the data is the same
assert np.allclose([np.sin(data).sum(axis=0)], [read_data[0]])