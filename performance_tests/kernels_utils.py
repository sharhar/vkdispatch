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
def k_const_warp(out: wp.array(dtype=float), mat1: wp.mat44f, mat2: wp.mat44f):
    i = wp.tid()
    if i == 0:
        out[i] = out[i] + wp.determinant(mat1) + wp.determinant(mat2)

@wp.kernel
def k_param_stream_warp(out: wp.array(dtype=float), matricies: wp.array(dtype=wp.mat44f), param_index: int):
    i = wp.tid()
    if i == 0:
        out[i] = out[i] + wp.determinant(matricies[param_index]) + wp.determinant(matricies[param_index + 1])

def make_graph_warp(kernel, out, matricies, batch_size, stream, device, do_streaming):
    identity_matrix = np.diag(np.ones(shape=(4,), dtype=np.float32))

    with wp.ScopedCapture(device=device, stream=stream) as capture:
        for i in range(batch_size):
            inputs = [out, identity_matrix, identity_matrix] if not do_streaming else [out, matricies, 2*i]

            wp.launch(
                kernel,
                dim=1,
                inputs=inputs,
                device=device,
                stream=stream
            )

    return capture.graph

def do_benchmark_warp(kernel, params_host, kernel_type, batch_size, iter_count, streams_per_device, stream_count, device_ids):
    out_arrays = []
    params_arrays = []
    h_buffs = []
    graphs = []
    streams = []

    devices = [wp.get_device(f"cuda:{device_id}") for device_id in device_ids]

    total_streams = streams_per_device * len(device_ids)

    for i in range(total_streams):
        device = devices[i % len(device_ids)]

        stream = wp.Stream(device=device)

        streams.append(stream)

        out_arrays.append(wp.zeros(shape=(1,), dtype=wp.float32, device=device))
        
        if kernel_type == "param_stream":
            h_buffs.append(wp.zeros(shape=(2 * batch_size,), dtype=wp.mat44f, device=device, pinned=True))
            params_arrays.append(wp.zeros(shape=(2 * batch_size,), dtype=wp.mat44f, device=device))
        else:
            h_buffs.append(None)
            params_arrays.append(None)

        graphs.append(make_graph_warp(
            kernel,
            out_arrays[i],
            params_arrays[i] ,
            batch_size,
            stream,
            device,
            kernel_type == "param_stream"
        ))

    assert iter_count % batch_size == 0, "iter_count must be a multiple of batch_size"

    num_graph_launches = iter_count // batch_size

    start_time = time.perf_counter()
    for i in range(num_graph_launches):
        device = devices[i % len(device_ids)]
        stream_idx = i % total_streams

        if kernel_type == "param_stream":
            h_buffs[stream_idx].numpy()[:] = params_host[2*i*batch_size:2*(i+1)*batch_size]
            wp.copy(params_arrays[stream_idx], h_buffs[stream_idx], stream=streams[stream_idx])

        wp.capture_launch(graphs[stream_idx], stream=streams[stream_idx])
    
    for dev in devices:
        wp.synchronize_device(dev)
    end_time = time.perf_counter()

    # Cleanup
    del graphs
    del streams
    del out_arrays
    del params_arrays
    
    if kernel_type == "param_stream":
        del h_buffs

    wp.synchronize_device("cuda:0")
    gc.collect()

    return end_time - start_time

# ----------- Define kernels for measuring launch overheads ---------------


@vd.shader(local_size=(1, 1, 1), workgroups=(1, 1, 1), enable_exec_bounds=False)
def k_const_vkdispatch(out: vc.Buff[vc.f32], mat1: vc.Const[vc.m4], mat2: vc.Const[vc.m4]):
    i = vc.global_invocation().x
    vc.if_statement(i == 0)
    out[i] = out[i] + vc.determinant(mat1) + vc.determinant(mat2)
    vc.end()

@vd.shader(local_size=(1, 1, 1), workgroups=(1, 1, 1), enable_exec_bounds=False)
def k_param_stream_vkdispatch(out: vc.Buff[vc.f32], mat1: vc.Var[vc.m4], mat2: vc.Var[vc.m4]):
    i = vc.global_invocation().x
    vc.if_statement(i == 0)
    out[i] = out[i] + vc.determinant(mat1) + vc.determinant(mat2)
    vc.end()

def do_benchmark_vkdispatch(kernel, params_host, kernel_type, batch_size, iter_count, streams_per_device, stream_count, device_ids):
    out_buff = vd.Buffer(shape=(1,), var_type=vd.float32)
    identity_matrix = np.diag(np.ones(shape=(4,), dtype=np.float32))

    do_streaming = kernel_type == "param_stream"

    graph = vd.CommandGraph()
    
    kernel(
        out_buff,
        graph.bind_var("mat1") if do_streaming else identity_matrix,
        graph.bind_var("mat2") if do_streaming else identity_matrix,
        graph=graph
    )

    assert iter_count % batch_size == 0, "iter_count must be a multiple of batch_size"

    num_graph_launches = iter_count // batch_size

    total_streams = streams_per_device * len(device_ids)

    vd.queue_wait_idle()   
    
    start_time = time.perf_counter()
    for i in range(num_graph_launches):
        if kernel_type == "param_stream":
            graph.set_var("mat1", params_host[2*i*batch_size:2*(i+1)*batch_size:2])
            graph.set_var("mat2", params_host[2*i*batch_size+1:2*(i+1)*batch_size:2])

        raw_stream_index = i % total_streams
        raw_stream_index = raw_stream_index + (stream_count - streams_per_device) * raw_stream_index // streams_per_device
        graph.submit(instance_count=batch_size, queue_index=raw_stream_index)

    vd.queue_wait_idle()   
    end_time = time.perf_counter()

    out_buff.destroy()
    graph.destroy()

    vd.queue_wait_idle()

    return end_time - start_time

kernels = {
    "warp": {
        "const": k_const_warp,
        "param_stream": k_param_stream_warp,
    },
    "vkdispatch": {
        "const": k_const_vkdispatch,
        "param_stream": k_param_stream_vkdispatch,
    }
}

benchmarks = {
    "warp": do_benchmark_warp,
    "vkdispatch": do_benchmark_vkdispatch
}

def do_benchmark(platform, kernel_type, params_host, batch_size, iter_count, streams_per_device, stream_count, device_ids):
    elapsed_time = benchmarks[platform](
        kernels[platform][kernel_type],
        params_host,
        kernel_type,
        batch_size,
        iter_count,
        streams_per_device,
        stream_count,
        device_ids
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