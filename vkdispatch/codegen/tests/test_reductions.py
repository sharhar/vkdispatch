import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *
import numpy as np

vd.make_context(use_cpu=True)

def test_reductions_sum():
    initial_buffer = vd.Buffer((1024,) , vd.float32)
    initial_buffer.write(np.random.rand(1024).astype(np.float32))

    print(initial_buffer.read(0).sum())
    
    # Create a buffer
    buf = vd.Buffer((1024,) , vd.float32)

    # Create a numpy array
    data = np.random.rand(1024).astype(np.float32)

    # Write the data to the buffer
    buf.write(data)

    @vc.map_reduce(
            exec_size=lambda args: args.buffer.size, 
            reduction=lambda x, y: x + y, 
            reduction_identity=0
    )
    def sum_map(ind: Const[i32], buffer: Buff[f32]) -> f32:
        return buffer[ind]
    
    res_buf = sum_map(buf)

    # Read the data from the buffer
    read_data = res_buf.read(0)

    # Check that the data is the same
    assert np.allclose([data.sum()], [read_data[0]])

def test_mapped_reductions():
    # Create a buffer
    buf = vd.Buffer((1024,) , vd.float32)

    # Create a numpy array
    data = np.random.rand(1024).astype(np.float32)

    # Write the data to the buffer
    buf.write(data)

    @vc.map_reduce(
            exec_size=lambda args: args.buffer.size, 
            reduction=lambda x, y: x + y, 
            reduction_identity=0
    )
    def sum_map(ind: Const[i32], buffer: Buff[f32]) -> f32:
        return vc.sin(buffer[ind])
    
    res_buf = sum_map(buf)

    # Read the data from the buffer
    read_data = res_buf.read(0)

    # Check that the data is the same
    assert np.allclose([np.sin(data).sum()], [read_data[0]])
