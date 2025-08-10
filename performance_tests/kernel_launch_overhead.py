import numpy as np
import vkdispatch as vd
import matplotlib.pyplot as plt


from kernel_launch_overhead_utils import do_benchmark, adjust_lightness

platforms = [
    "warp",
    "vkdispatch"
]

kernel_types = [
    #"noop",
    "const",
    "param_stream",
    "param_pointer_chase"
]

# ----------- Define kernels dictionary -----------------------------------

# Assign base colors for each platform
platform_colors = {
    platform: plt.cm.tab10(i % 10)  # tab10 colormap cycles nicely
    for i, platform in enumerate(platforms)
}

# Kernel lightness factors
kernel_factors = {
    kernel_type: 0.50 + 1 * (i / max(1, len(kernel_types) - 1))
    for i, kernel_type in enumerate(kernel_types)
}

stream_count = 2

vd.make_context(devices=[0], queue_families=[vd.select_queue_families(0, stream_count)])

print(vd.get_context().queue_families)

means = {platform: {kernel_type: [] for kernel_type in kernel_types} for platform in platforms}
stds = {platform: {kernel_type: [] for kernel_type in kernel_types} for platform in platforms}

iter_count = 1024 * 1024  # Total number of iterations for the benchmark
run_count = 2  # Number of times to run each benchmark

params_host = np.arange(iter_count, dtype=np.float32)

batch_size_exponents = list(range(3, 11))  # Batch sizes from 8 to 1024

for batch_size_exp in batch_size_exponents:
    batch_size = 2 ** batch_size_exp

    for platform in platforms:

        for kernel_type in kernel_types:
            
            rates = []
            for i in range(run_count):
                print(f"Benchmarking {kernel_type} kernel with batch size {batch_size} on {platform} Run {i + 1}/{run_count}...")
                rates.append(do_benchmark(
                    platform,
                    kernel_type,
                    params_host,
                    batch_size,
                    iter_count,
                    stream_count
                ))

            mean_rate = np.mean(rates)
            std_rate = np.std(rates)

            means[platform][kernel_type].append(mean_rate)
            stds[platform][kernel_type].append(std_rate)

# ----------- Print results ------------------------------------------------

print("\nBenchmark Results:")
for platform in platforms:
    print(f"\nPlatform: {platform}")
    for kernel_type in kernel_types:
        print(f"\nKernel Type: {kernel_type}")
        for batch_size_exp, (mean, std) in enumerate(zip(means[platform][kernel_type], stds[platform][kernel_type])):
            batch_size = 2 ** (batch_size_exponents[batch_size_exp])
            print(f"Batch Size: {batch_size}, Mean Kernels/second: {mean:.6f} s, Std Dev: {std:.6f} s")
print("\nBenchmark completed.")

# ----------- Plot results (optional) -----------------------------

plt.figure(figsize=(10, 6))
for platform in platforms:
    for kernel_type in kernel_types:
        base_color = platform_colors[platform]
        color = adjust_lightness(base_color, kernel_factors[kernel_type])

        plt.errorbar(
            [2 ** (batch_size_exponents[i]) for i in range(len(means[platform][kernel_type]))],
            means[platform][kernel_type],
            yerr=stds[platform][kernel_type],
            label=f"{platform} - {kernel_type}",
            capsize=5,
            color=color
        )

plt.xscale('log', base=2)
#plt.yscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Kernels/second')
plt.title(f'Kernel Launch Overhead Benchmark (Stream Count: {stream_count})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'kernel_launch_overhead_{stream_count}_streams.png')
