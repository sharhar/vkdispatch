import warp as wp
import time
import gc
import numpy as np

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
        param_index = index_buffer[0]
        out[i] = params[param_index] + 1.0
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
    out_array = wp.empty(shape=(batch_size,), dtype=wp.float32, device="cuda:0")
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


# ----------- Define platforms and kernel types ----------------------------

platforms = [
    "warp"
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
    }
}

means = {kernel_type: [] for kernel_type in kernel_types}
stds = {kernel_type: [] for kernel_type in kernel_types}

iter_count = 1024 * 1024  # Total number of iterations for the benchmark
run_count = 3  # Number of times to run each benchmark

for batch_size_exp in range(3, 10):
    batch_size = 2 ** batch_size_exp

    for kernel_type in kernel_types:
        kernel = kernels["warp"][kernel_type]
        iter_count = 1024 * 1024
        params_host = np.arange(iter_count, dtype=np.float32)
        
        times = []

        for i in range(run_count):
            print(f"Benchmarking {kernel_type} kernel with batch size {batch_size} on Warp Run {i + 1}/{run_count}...")
            elapsed_time = do_benchmark_warp(kernel, params_host, kernel_type, batch_size, iter_count)
            times.append(elapsed_time)

        mean_time = np.mean(times)
        std_time = np.std(times)

        means[kernel_type].append(mean_time)
        stds[kernel_type].append(std_time)

# ----------- Print results ------------------------------------------------
print("\nBenchmark Results:")
for kernel_type in kernel_types:
    print(f"\nKernel Type: {kernel_type}")
    for batch_size_exp, (mean, std) in enumerate(zip(means[kernel_type], stds[kernel_type])):
        batch_size = 2 ** (batch_size_exp + 1)
        print(f"Batch Size: {batch_size}, Mean Time: {mean:.6f} s, Std Dev: {std:.6f} s")
print("\nBenchmark completed.")

# ----------- Plot results (optional) -----------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for kernel_type in kernel_types:
    plt.errorbar(
        [2 ** (i + 1) for i in range(len(means[kernel_type]))],
        means[kernel_type],
        yerr=stds[kernel_type],
        label=kernel_type,
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


