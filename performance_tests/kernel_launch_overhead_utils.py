import warp as wp
import time
import gc
import numpy as np
import vkdispatch as vd
import vkdispatch.codegen as vc
import matplotlib.colors as mcolors
import colorsys

# ----------- Define kernels for measuring launch overheads ---------------

@wp.kernel
def k_noop_warp(out: wp.array(dtype=float), params: wp.array(dtype=float), index_buffer: wp.array(dtype=int), param_index: int):
    pass

@wp.kernel
def k_const_warp(out: wp.array(dtype=float), params: wp.array(dtype=float), index_buffer: wp.array(dtype=int), param_index: int):
    i = wp.tid()
    if i == 0:
        out[i] = out[i] + 1.0

@wp.kernel
def k_param_stream_warp(out: wp.array(dtype=float), params: wp.array(dtype=float), index_buffer: wp.array(dtype=int), param_index: int):
    i = wp.tid()
    if i == 0:
        out[i] = params[param_index] + 1.0

@wp.kernel
def k_param_pointer_chase_warp(out: wp.array(dtype=float), params: wp.array(dtype=float), index_buffer: wp.array(dtype=int), param_index: int):
    i = wp.tid()
    if i == 0:
        param_idx = index_buffer[0]
        out[i] = params[param_idx] + 1.0
        index_buffer[0] += 1

def make_graph_warp(kernel, out, params, index_buffer, batch_size, stream):
    with wp.ScopedCapture(device="cuda:0", stream=stream) as capture:
        for i in range(batch_size):
            wp.launch(
                kernel,
                dim=1,
                inputs=[out, params, index_buffer, i],
                device=wp.get_device(),
                stream=stream
            )

    return capture.graph

def do_benchmark_warp(kernel, params_host, kernel_type, batch_size, iter_count, stream_count):
    out_arrays = []
    index_buffers = []
    params_arrays = []
    h_buffs = []
    graphs = []
    streams = []

    for i in range(stream_count):
        stream = wp.Stream(device=wp.get_device())

        streams.append(stream)

        out_arrays.append(wp.empty(shape=(1,), dtype=wp.float32, device="cuda:0"))
        index_buffers.append(wp.empty(shape=(1,), dtype=wp.int32, device="cuda:0"))
        
        if kernel_type == "param_stream":
            h_buffs.append(wp.empty(shape=(batch_size,), dtype=wp.float32, device="cuda:0", pinned=True))
            params_arrays.append(wp.empty(shape=(batch_size,), dtype=wp.float32, device="cuda:0"))
        else:
            params_arrays.append(wp.array(params_host, dtype=wp.float32, device="cuda:0"))

        graphs.append(make_graph_warp(
            kernel,
            out_arrays[i],
            params_arrays[i],
            index_buffers[i],
            batch_size,
            stream)
        )

    assert iter_count % batch_size == 0, "iter_count must be a multiple of batch_size"

    num_graph_launches = iter_count // batch_size

    start_time = time.perf_counter()
    for i in range(num_graph_launches):
        for j in range(stream_count):

            if kernel_type == "param_stream":
                h_buffs[j].numpy()[:] = params_host[i*batch_size:(i+1)*batch_size]
                wp.copy(params_arrays[j], h_buffs[j], stream=streams[j])

            wp.capture_launch(graphs[j], stream=streams[j])
        
    wp.synchronize_device("cuda:0")
    end_time = time.perf_counter()

    # Cleanup
    del graphs
    del streams
    del out_arrays
    del params_arrays
    del index_buffers
    
    if kernel_type == "param_stream":
        del h_buffs

    wp.synchronize_device("cuda:0")
    gc.collect()

    return end_time - start_time

# ----------- Define kernels for measuring launch overheads ---------------

@vd.shader(local_size=(1, 1, 1), workgroups=(1, 1, 1), enable_exec_bounds=False)
def k_noop_vkdispatch(out: vc.Buff[vc.f32], params: vc.Buff[vc.f32], index_buffer: vc.Buff[vc.i32]):
    pass

@vd.shader(local_size=(1, 1, 1), workgroups=(1, 1, 1), enable_exec_bounds=False)
def k_const_vkdispatch(out: vc.Buff[vc.f32], params: vc.Buff[vc.f32], index_buffer: vc.Buff[vc.i32]):
    i = vc.global_invocation().x
    vc.if_statement(i == 0)
    out[i] = out[i] + 1.0
    vc.end()

@vd.shader(local_size=(1, 1, 1), workgroups=(1, 1, 1), enable_exec_bounds=False)
def k_param_stream_vkdispatch(out: vc.Buff[vc.f32], param: vc.Var[vc.f32], index_buffer: vc.Buff[vc.i32]):
    i = vc.global_invocation().x
    vc.if_statement(i == 0)
    out[i] = param + 1.0
    vc.end()

@vd.shader(local_size=(1, 1, 1), workgroups=(1, 1, 1))
def k_param_pointer_chase_vkdispatch(out: vc.Buff[vc.f32], params: vc.Buff[vc.f32], index_buffer: vc.Buff[vc.i32]):
    i = vc.global_invocation().x
    vc.if_statement(i == 0)
    param_idx = index_buffer[0]
    out[i] = params[param_idx] + 1.0
    index_buffer[0] += 1
    vc.end()

def do_benchmark_vkdispatch(kernel, params_host, kernel_type, batch_size, iter_count, stream_count):
    out_buff = vd.Buffer(shape=(1,), var_type=vd.float32)
    index_buffer = vd.Buffer(shape=(1,), var_type=vd.int32)
    index_buffer.write(np.array([0], dtype=np.int32))
    params_buff = vd.Buffer(shape=(iter_count,), var_type=vd.float32)
    params_buff.write(params_host)

    cmd_stream = vd.CommandStream()
    
    kernel(
        out_buff,
        cmd_stream.bind_var("param") if kernel_type == "param_stream" else params_buff,
        index_buffer,
        cmd_stream=cmd_stream
    )

    assert iter_count % batch_size == 0, "iter_count must be a multiple of batch_size"

    num_graph_launches = iter_count // batch_size

    start_time = time.perf_counter()
    for i in range(num_graph_launches):
        if kernel_type == "param_stream":
            cmd_stream.set_var("param", params_host[i*batch_size:(i+1)*batch_size])

        cmd_stream.submit(instance_count=batch_size, stream_index= i % stream_count)
        
    out_buff.read(0)
    end_time = time.perf_counter()

    return end_time - start_time

kernels = {
    "warp": {
        "noop": k_noop_warp,
        "const": k_const_warp,
        "param_stream": k_param_stream_warp,
        "param_pointer_chase": k_param_pointer_chase_warp
    },
    "vkdispatch": {
        "noop": k_noop_vkdispatch,
        "const": k_const_vkdispatch,
        "param_stream": k_param_stream_vkdispatch,
        "param_pointer_chase": k_param_pointer_chase_vkdispatch
    }
}

benchmarks = {
    "warp": do_benchmark_warp,
    "vkdispatch": do_benchmark_vkdispatch
}

def do_benchmark(platform, kernel_type, params_host, batch_size, iter_count, stream_count):
    elapsed_time = benchmarks[platform](
        kernels[platform][kernel_type],
        params_host,
        kernel_type,
        batch_size,
        iter_count,
        stream_count
    )

    return iter_count / elapsed_time

def adjust_lightness(color, factor):
    """Lighten or darken a given matplotlib color by multiplying its lightness by 'factor'."""
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    r, g, b = mcolors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)