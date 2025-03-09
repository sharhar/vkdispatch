import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *
import numpy as np

#vd.initialize(debug_mode=True)
#vd.make_context(devices=[2])

vd.make_context(use_cpu=True)

def test_reductions_sum():
    # Create a buffer
    buf = vd.Buffer((1536,) , vd.float32)

    # Create a numpy array
    data = np.random.rand(1536).astype(np.float32)

    # Write the data to the buffer
    buf.write(data)

    @vd.map_reduce(
            exec_size=lambda args: args.buffer.size,
            group_size=512,
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

    @vd.map_reduce(
            exec_size=lambda args: args.buffer.size,
            group_size=512,
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


def test_listed_reductions():
    # Create a buffer
    buf = vd.Buffer((1024,) , v2)
    buf2 = vd.Buffer((1,) , v2)

    # Create a numpy array
    data = np.random.rand(1024, 2).astype(np.float32)

    # Write the data to the buffer
    buf.write(data)

    @vd.map_reduce(
            exec_size=lambda args: args.buffer.size,
            group_size=512,
            reduction=lambda x, y: x + y, 
            reduction_identity=0
    )
    def sum_map(ind: Const[i32], buffer: Buff[v2]) -> v2:
        return vc.sin(buffer[ind])
    
    cmd_stream = vd.CommandStream()

    old_list = vd.set_global_cmd_stream(cmd_stream)

    res_buf = sum_map(buf, cmd_stream=cmd_stream)

    vd.set_global_cmd_stream(old_list)

    cmd_stream.submit()

    # Read the data from the buffer
    read_data = res_buf.read(0)

    # Check that the data is the same
    assert np.allclose([np.sin(data).sum(axis=0)], [read_data[0]])

def test_pure_reductions():
    # Create a buffer

    data_size = 345000

    #buf = vd.Buffer((data_size,) , vd.float32)

    # Create a numpy array
    data = np.random.rand(data_size).astype(np.float32)

    # Write the data to the buffer
    buf = vd.asbuffer(data)

    @vd.reduce(0)
    def sum_reduce(a: f32, b: f32) -> f32:
        result = (a + b).copy()
        return result

    res_buf = sum_reduce(buf)

    # Read the data from the buffer
    read_data = res_buf.read(0)

    # Check that the data is the same
    assert np.allclose([data.sum()], [read_data[0]])