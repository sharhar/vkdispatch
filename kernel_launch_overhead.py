import warp as wp
import time
import gc
import numpy as np
import vkdispatch as vd
import vkdispatch.codegen as vc

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

def make_graph_warp(kernel, out, params, index_buffer, batch_size):
    with wp.ScopedCapture(device="cuda:0") as capture:
        for i in range(batch_size):
            wp.launch(
                kernel,
                dim=1,
                inputs=[out, params, index_buffer, i],
                device=wp.get_device()
            )

    return capture.graph

def do_benchmark_warp(kernel, params_host, kernel_type, batch_size, iter_count):
    out_array = wp.empty(shape=(1,), dtype=wp.float32, device="cuda:0")
    if kernel_type == "param_stream":
        params_array = wp.empty(shape=(batch_size,), dtype=wp.float32, device="cuda:0")
        h_buff = wp.empty(shape=(batch_size,), dtype=wp.float32, device="cuda:0", pinned=True)
    else:
        params_array = wp.array(params_host, dtype=wp.float32, device="cuda:0")
    index_buffer = wp.empty(shape=(1,), dtype=wp.int32, device="cuda:0")

    graph = make_graph_warp(kernel, out_array, params_array, index_buffer, batch_size)

    assert iter_count % batch_size == 0, "iter_count must be a multiple of batch_size"

    num_graph_launches = iter_count // batch_size

    start_time = time.perf_counter()
    for i in range(num_graph_launches):
        if kernel_type == "param_stream":
            h_buff.numpy()[:] = params_host[i*batch_size:(i+1)*batch_size]
            wp.copy(params_array, h_buff)

        wp.capture_launch(graph)
    wp.synchronize_device("cuda:0")
    end_time = time.perf_counter()

    # Cleanup
    del graph
    del out_array
    del params_array
    del index_buffer
    
    if kernel_type == "param_stream":
        del h_buff

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

def do_benchmark_warp(kernel, params_host, kernel_type, batch_size, iter_count):
    out_array = wp.empty(shape=(1,), dtype=wp.float32, device="cuda:0")
    if kernel_type == "param_stream":
        params_array = wp.empty(shape=(batch_size,), dtype=wp.float32, device="cuda:0")
        h_buff = wp.empty(shape=(batch_size,), dtype=wp.float32, device="cuda:0", pinned=True)
    else:
        params_array = wp.array(params_host, dtype=wp.float32, device="cuda:0")
    index_buffer = wp.empty(shape=(1,), dtype=wp.int32, device="cuda:0")

    graph = make_graph_warp(kernel, out_array, params_array, index_buffer, batch_size)

    assert iter_count % batch_size == 0, "iter_count must be a multiple of batch_size"

    num_graph_launches = iter_count // batch_size

    start_time = time.perf_counter()
    for i in range(num_graph_launches):
        if kernel_type == "param_stream":
            h_buff.numpy()[:] = params_host[i*batch_size:(i+1)*batch_size]
            wp.copy(params_array, h_buff)

        wp.capture_launch(graph)
    wp.synchronize_device("cuda:0")
    end_time = time.perf_counter()

    # Cleanup
    del graph
    del out_array
    del params_array
    del index_buffer
    
    if kernel_type == "param_stream":
        del h_buff

    wp.synchronize_device("cuda:0")
    gc.collect()

    return end_time - start_time

def do_benchmark_vkdispatch(kernel, params_host, kernel_type, batch_size, iter_count):
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

        cmd_stream.submit_any(instance_count=batch_size)
        
    out_buff.read(0)
    end_time = time.perf_counter()

    return end_time - start_time


# ----------- Define platforms and kernel types ----------------------------

platforms = [
    "warp",
    "vkdispatch"
]

kernel_types = [
    "noop",
    "const",
    "param_stream",
    "param_pointer_chase"
]

# ----------- Define kernels dictionary -----------------------------------

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

vd.make_context()

means = {platform: {kernel_type: [] for kernel_type in kernel_types} for platform in platforms}
stds = {platform: {kernel_type: [] for kernel_type in kernel_types} for platform in platforms}

iter_count = 1024 * 1024  # Total number of iterations for the benchmark
run_count = 5  # Number of times to run each benchmark

for batch_size_exp in range(3, 12):
    batch_size = 2 ** batch_size_exp

    for platform in platforms:

        for kernel_type in kernel_types:
            kernel = kernels[platform][kernel_type]
            iter_count = 1024 * 1024
            params_host = np.arange(iter_count, dtype=np.float32)
            
            times = []

            for i in range(run_count):
                print(f"Benchmarking {kernel_type} kernel with batch size {batch_size} on {platform} Run {i + 1}/{run_count}...")
                elapsed_time = benchmarks[platform](kernel, params_host, kernel_type, batch_size, iter_count)
                times.append(elapsed_time)

            mean_time = np.mean(times)
            std_time = np.std(times)

            means[platform][kernel_type].append(mean_time)
            stds[platform][kernel_type].append(std_time)

# ----------- Print results ------------------------------------------------

print("\nBenchmark Results:")
for platform in platforms:
    print(f"\nPlatform: {platform}")
    for kernel_type in kernel_types:
        print(f"\nKernel Type: {kernel_type}")
        for batch_size_exp, (mean, std) in enumerate(zip(means[platform][kernel_type], stds[platform][kernel_type])):
            batch_size = 2 ** (batch_size_exp + 1)
            print(f"Batch Size: {batch_size}, Mean Time: {mean:.6f} s, Std Dev: {std:.6f} s")
print("\nBenchmark completed.")

# ----------- Plot results (optional) -----------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for platform in platforms:
    for kernel_type in kernel_types:
        plt.errorbar(
            [2 ** (i + 1) for i in range(len(means[platform][kernel_type]))],
            means[platform][kernel_type],
            yerr=stds[platform][kernel_type],
            label=f"{platform} - {kernel_type}",
            capsize=5
        )
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Mean Time (s)')
plt.title('Kernel Launch Overhead Benchmark')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('kernel_launch_overhead.png')