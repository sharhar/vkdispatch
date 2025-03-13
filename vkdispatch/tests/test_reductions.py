import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *
import numpy as np

vd.initialize(debug_mode=True)
#vd.make_context(devices=[2])

vd.make_context(use_cpu=True)

def test_reductions_sum():
    # Create a buffer
    buf = vd.Buffer((1536,) , vd.float32)

    # Create a numpy array
    data = np.random.rand(1536).astype(np.float32)

    # Write the data to the buffer
    buf.write(data)

    @vd.map_reduce(vd.SubgroupAdd)
    def sum_map(buffer: Buff[f32]) -> f32:
        return buffer[vc.mapping_index()]

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

    @vd.map_reduce(vd.SubgroupAdd)
    def sum_map(buffer: Buff[f32]) -> f32:
        return vc.sin(buffer[vc.mapping_index()])
    
    res_buf = sum_map(buf)

    # Read the data from the buffer
    read_data = res_buf.read(0)

    # Check that the data is the same
    assert np.allclose([np.sin(data).sum()], [read_data[0]])

def test_listed_reductions():
    # Create a buffer
    buf = vd.Buffer((1024,) , v2)
    buf2 = vd.Buffer((1024,) , v2)

    # Create a numpy array
    data = np.random.rand(1024, 2).astype(np.float32)
    data2 = np.random.rand(1024, 2).astype(np.float32)

    # Write the data to the buffer
    buf.write(data)
    buf2.write(data2)

    @vd.map_reduce(vd.SubgroupAdd)
    def sum_map(buffer: Buff[v2], buffer2: Buff[v2]) -> v2:
        ind = vc.mapping_index()

        return vc.sin(buffer[ind] + buffer2[ind])
    
    cmd_stream = vd.CommandStream()

    old_list = vd.set_global_cmd_stream(cmd_stream)

    res_buf = sum_map(buf, buf2, cmd_stream=cmd_stream)

    vd.set_global_cmd_stream(old_list)

    cmd_stream.submit()

    # Read the data from the buffer
    read_data = res_buf.read(0)

    # Check that the data is the same
    assert np.allclose([np.sin(data + data2).sum(axis=0)], [read_data[0]])

def test_pure_reductions():
    # Create a buffer

    data_size = 300000

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
    difference = data.sum(dtype=np.float32) - read_data[0]

    assert np.abs(difference / data.sum(dtype=np.float32)) < 1e-3

def test_batched_mapped_reductions():
    batch_size = 10
    data_size = 300000

    # Create a numpy array
    data = np.random.rand(batch_size, data_size).astype(np.float32)

    # Write the data to the buffer
    buf = vd.asbuffer(data)

    @vd.map_reduce(vd.SubgroupAdd, axes=[1])
    def sum_map(buffer: Buff[f32]) -> f32:
        return vc.sin(buffer[vc.mapping_index()])
    
    res_buf = sum_map(buf)

    # Read the data from the buffer
    read_data = res_buf.read(0)[0]

    # Check that the data is the same
    assert np.allclose([np.sin(data).sum(axis=1)], [read_data])
