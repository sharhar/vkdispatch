import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import *
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

    @vd.reduce.map_reduce(vd.reduce.SubgroupAdd)
    def sum_map(buffer: Buff[f32]) -> f32:
        return buffer[vd.reduce.mapped_io_index()]

    res_buf = sum_map(buf)

    # Read the data from the buffer
    read_data = res_buf.read(0)

    # Check that the data is the same
    assert np.allclose([data.sum()], [read_data[0]])
"""
def test_mapped_reductions():
    # Create a buffer
    buf = vd.Buffer((1024,) , vd.float32)

    # Create a numpy array
    data = np.random.rand(1024).astype(np.float32)

    # Write the data to the buffer
    buf.write(data)

    @vd.reduce.map_reduce(vd.reduce.SubgroupAdd)
    def sum_map(buffer: Buff[f32]) -> f32:
        return vc.sin(buffer[vd.reduce.mapped_io_index()])
    
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

    @vd.reduce.map_reduce(vd.reduce.SubgroupAdd)
    def sum_map(buffer: Buff[v2], buffer2: Buff[v2]) -> v2:
        ind = vd.reduce.mapped_io_index()
        return vc.sin(buffer[ind] + buffer2[ind])

    graph = vd.CommandGraph()

    old_graph = vd.set_global_graph(graph)
    res_buf = sum_map(buf, buf2, graph=graph)
    vd.set_global_graph(old_graph)

    graph.submit()

    vd.queue_wait_idle()

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

    @vd.reduce.reduce(0)
    def sum_reduce(a: f32, b: f32) -> f32:
        return a + b

    res_buf = sum_reduce(buf)

    # Read the data from the buffer
    read_data = res_buf.read(0)

    # Check that the data is the same
    difference = data.sum(dtype=np.float32) - read_data[0]

    assert np.abs(difference / data.sum(dtype=np.float32)) < 1e-3

def test_pure_reductions_with_mapping_function():
    # Create a buffer

    data_size = 300000

    # Create a numpy array
    data = np.random.rand(data_size).astype(np.float32)

    # Write the data to the buffer
    buf = vd.asbuffer(data)

    @vd.map
    def reduction_map(input: Buff[f32]) -> f32:
        return vc.sin(input[vd.reduce.mapped_io_index()])

    @vd.reduce.reduce(0, mapping_function=reduction_map)
    def sum_reduce(a: f32, b: f32) -> f32:
        return a + b

    res_buf = sum_reduce(buf)

    # Read the data from the buffer
    read_data = res_buf.read(0)

    # Check that the data is the same
    difference = np.sin(data).sum(dtype=np.float32) - read_data[0]

    assert np.abs(difference / data.sum(dtype=np.float32)) < 1e-3

def test_batched_mapped_reductions():
    batch_size = 10
    data_size = 300000

    # Create a numpy array
    data = np.random.rand(batch_size, data_size).astype(np.float32)

    # Write the data to the buffer
    buf = vd.asbuffer(data)

    @vd.reduce.map_reduce(vd.reduce.SubgroupAdd, axes=[1])
    def sum_map(buffer: Buff[f32]) -> f32:
        return vc.sin(buffer[vd.reduce.mapped_io_index()])
    
    res_buf = sum_map(buf)

    # Read the data from the buffer
    read_data = res_buf.read(0)[0]

    # Check that the data is the same
    assert np.allclose([np.sin(data).sum(axis=1)], [read_data])

def test_mapped_reductions_min():
    # Create a buffer
    buf = vd.Buffer((1024,), vd.float32)

    # Create a numpy array
    data = np.random.randn(1024).astype(np.float32)

    # Write the data to the buffer
    buf.write(data)

    @vd.reduce.map_reduce(vd.reduce.SubgroupMin)
    def min_map(buffer: Buff[f32]) -> f32:
        return buffer[vd.reduce.mapped_io_index()]

    res_buf = min_map(buf)

    # Read the data from the buffer
    read_data = res_buf.read(0)

    # Check that the data is the same
    assert np.allclose([data.min()], [read_data[0]])

def test_mapped_reductions_max():
    # Create a buffer
    buf = vd.Buffer((1024,), vd.float32)

    # Create a numpy array
    data = np.random.randn(1024).astype(np.float32)

    # Write the data to the buffer
    buf.write(data)

    @vd.reduce.map_reduce(vd.reduce.SubgroupMax)
    def max_map(buffer: Buff[f32]) -> f32:
        return buffer[vd.reduce.mapped_io_index()]

    res_buf = max_map(buf)

    # Read the data from the buffer
    read_data = res_buf.read(0)

    # Check that the data is the same
    assert np.allclose([data.max()], [read_data[0]])

def test_min_max_codegen_stage_creation():
    @vd.reduce.map_reduce(vd.reduce.SubgroupMin)
    def min_map(buffer: Buff[f32]) -> f32:
        return buffer[vd.reduce.mapped_io_index()]

    @vd.reduce.map_reduce(vd.reduce.SubgroupMax)
    def max_map(buffer: Buff[f32]) -> f32:
        return buffer[vd.reduce.mapped_io_index()]

    min_src_stage1, min_src_stage2 = min_map.get_src()
    max_src_stage1, max_src_stage2 = max_map.get_src()

    assert min_src_stage1 and min_src_stage2
    assert max_src_stage1 and max_src_stage2
"""