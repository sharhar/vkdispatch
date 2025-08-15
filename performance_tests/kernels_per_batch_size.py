import numpy as np
import vkdispatch as vd
import matplotlib.pyplot as plt
import sys
import time
import csv

from kernels_utils import do_benchmark, adjust_lightness

platforms = [
    "warp",
    "vkdispatch"
]

kernel_types = [
    "const",
    "param_stream",
]

test_configs = [
    ("warp", "const"),
    ("warp", "param_stream"),

    ("vkdispatch", "const"),
    ("vkdispatch", "param_stream"),
]


# ----------- Define kernels dictionary -----------------------------------

# Assign base colors for each platform
platform_colors = {
    platform: plt.cm.tab10(i % 10)  # tab10 colormap cycles nicely
    for i, platform in enumerate(platforms)
}

# Kernel lightness factors
kernel_factors = {
    kernel_type: 0.50 + 0.5 * (i / max(1, len(kernel_types) - 1))
    for i, kernel_type in enumerate(kernel_types)
}

stream_count = int(sys.argv[1])
device_ids = list(range(int(sys.argv[2])))

vkdispatch_queue_families = []

for device_id in device_ids:
    vkdispatch_queue_families.append(vd.select_queue_families(device_id, stream_count))

vd.make_context(devices=device_ids, queue_families=vkdispatch_queue_families)

datas = {platform: {kernel_type: [] for kernel_type in kernel_types} for platform in platforms}

iter_count = 4 * 1024 * 1024  # Total number of iterations for the benchmark
run_count = 10 # Number of times to run each benchmark

identity_matrix = np.diag(np.ones(shape=(4,), dtype=np.float32))

params_host = np.zeros(shape=(2*iter_count, 4, 4), dtype=np.float32)
params_host[:] = identity_matrix

batch_size_exponents = list(range(2, 14))  # Batch sizes from 8 to 1024

for batch_size_exp in batch_size_exponents:
    batch_size = 2 ** batch_size_exp

    for platform, kernel_type in test_configs:
        rates = []
        for i in range(run_count):
            print(f"Benchmarking {kernel_type} kernel with batch size {batch_size} on {platform} Run {i + 1}/{run_count}...")
            time.sleep(0.25)  # Simulate some delay before starting the benchmark
            rates.append(do_benchmark(
                platform,
                kernel_type,
                params_host,
                batch_size,
                iter_count,
                stream_count,
                stream_count,
                device_ids
            ))

        datas[platform][kernel_type].append(rates)

# ----------- Print results ------------------------------------------------

output_name = f"kernels_per_batch_size_{len(device_ids)}_devices_{stream_count}_streams"

with open(output_name + ".csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['Platform', 'Kernel Type', 'Batch Size'] + [f'Run {i + 1} (Kernels/second)' for i in range(run_count)] + ['Mean', 'Std Dev'])
    for platform, kernel_type in test_configs:
        test_data = datas[platform][kernel_type]
        for batch_size_idx, rates in enumerate(test_data):
            batch_size = 2 ** batch_size_exponents[batch_size_idx]
            
            rounded_rates = [int(round(rate, 0)) for rate in rates]
            rounded_mean = round(np.mean(rates), 2)
            rounded_std = round(np.std(rates), 2)
            
            writer.writerow([platform, kernel_type, batch_size] + rounded_rates + [rounded_mean, rounded_std])
print(f"Raw benchmark data written to {output_name}.csv")


# ----------- Plot results (optional) -----------------------------

plt.figure(figsize=(10, 6))
for platform, kernel_type in test_configs:
    base_color = platform_colors[platform]
    color = adjust_lightness(base_color, kernel_factors[kernel_type])

    test_data = datas[platform][kernel_type]

    means = [np.mean(data) for data in test_data]
    stds = [np.std(data) for data in test_data]

    plt.errorbar(
        [2 ** (batch_size_exponents[i]) for i in range(len(means))],
        means,
        yerr=stds,
        label=f"{platform} - {kernel_type}",
        capsize=5,
        color=color
    )

plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Batch Size')
plt.ylabel('Kernels/second')
plt.title(f'Kernel Launch Overhead Benchmark (Stream Count: {stream_count}, Devices: {len(device_ids)}, Param Size: 128 bytes)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_name + "_log.png")

plt.yscale('linear')
plt.savefig(output_name + "_linear.png")
