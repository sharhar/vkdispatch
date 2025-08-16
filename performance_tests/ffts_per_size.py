from ffts_utils import run_benckmark
import numpy as np
import csv
import sys

data_size = 16 * 1024 * 1024 # 16 MB

test_backends = ['torch', 'vkdispatch', 'vkfft', 'scipy']

axis = int(sys.argv[1])
iter_count = 1000
iter_batch = 100
run_count = 3

fft_sizes = [2**i for i in range(1, 13)]  # FFT sizes from 4 to 16384
batched_axis = (axis + 1) % 2

data = {backend_name: [] for backend_name in test_backends}

for ii, fft_size in enumerate(fft_sizes):
    shape = [0, 0]

    shape[axis] = fft_size
    shape[batched_axis] = data_size // fft_size

    for backend_name in test_backends:
        rates = []

        print(f"Running {backend_name} backend for FFT size {fft_size}...")

        for _ in range(run_count):
            actual_iter_count = iter_count

            if backend_name == 'scipy':
                actual_iter_count //= 25  # Reduce iterations for scipy to avoid long execution time

            gb_per_second = run_benckmark(backend_name, shape, axis, actual_iter_count, iter_batch)
            print(f"Run {_ + 1}/{run_count}: {gb_per_second:.2f} GB/s")
            rates.append(gb_per_second)

        data[backend_name].append(rates)

# ----------- Print results ------------------------------------------------

output_name = f"ffts_per_size_{axis}_axis"
with open(output_name + ".csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['Backend', 'FFT Size'] + [f'Run {i + 1} (GB/s)' for i in range(run_count)] + ['Mean', 'Std Dev'])
    
    for i, fft_size in enumerate(fft_sizes):
        for backend_name in test_backends:
            runs_data = data[backend_name][i]

            rounded_data = [round(rate, 2) for rate in runs_data]
            rounded_mean = round(np.mean(runs_data), 2)
            rounded_std = round(np.std(runs_data), 2)

            writer.writerow([fft_size, backend_name] + rounded_data + [rounded_mean, rounded_std])

print(f"Results saved to {output_name}.csv")


# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for backend_name in test_backends:
    means = [np.mean(data[backend_name][i]) for i in range(len(fft_sizes))]
    stds = [np.std(data[backend_name][i]) for i in range(len(fft_sizes))]
    
    plt.errorbar(
        fft_sizes,
        means,
        yerr=stds,
        label=backend_name,
        capsize=5,
    )
plt.xscale('log', base=2)
plt.xlabel('FFT Size')
plt.ylabel('GB/s')
plt.title('FFT Performance Comparison')
plt.legend()
plt.grid(True)
plt.savefig(f"{output_name}.png")