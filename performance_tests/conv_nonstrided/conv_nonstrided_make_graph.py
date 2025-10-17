import glob
import csv
from typing import Dict, Tuple, Set, List
from matplotlib import pyplot as plt
import numpy as np

# Nested structure:
# merged[backend][fft_size] = (mean, std)
MergedType = Dict[str, Dict[int, Tuple[float, float]]]

def read_bench_csvs() -> Tuple[MergedType, Set[str], Set[int]]:
    pattern = 'conv_nonstrided_*.csv'
    files = glob.glob(pattern)

    merged: MergedType = {}
    backends: Set[str] = set()
    fft_sizes: Set[int] = set()

    for filename in files:
        print(f'Reading: {filename}')
        with open(filename, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                backend = row['Backend'].strip()
                size = int(row['FFT Size'])
                mean = float(row['Mean'])
                std = float(row['Std Dev'])

                backends.add(backend)
                fft_sizes.add(size)

                if backend not in merged:
                    merged[backend] = {}

                # last one wins if duplicates appear across files
                merged[backend][size] = (mean, std)

    return merged, backends, fft_sizes

def save_grouped_bar_graph(backends: List[str],
                           fft_sizes: List[int],
                           merged: MergedType,
                           min_fft_size: int = None,
                           outfile: str = 'conv_graph.png'):
    # Choose the sizes to display
    used_fft_sizes = [s for s in sorted(fft_sizes) if (min_fft_size is None or s >= min_fft_size)]
    if not used_fft_sizes:
        print('No FFT sizes to plot after filtering.')
        return

    x = np.arange(len(used_fft_sizes), dtype=float)
    n_backends = max(1, len(backends))
    width = 0.8 / n_backends  # total group width ~0.8

    plt.figure(figsize=(12, 6))

    for j, backend in enumerate(backends):
        # Center bars around tick: offsets in [-0.5..+0.5]*group_width
        xj = x + (j - (n_backends - 1) / 2) * width

        xs, heights, errs = [], [], []
        for i, size in enumerate(used_fft_sizes):
            entry = merged.get(backend, {}).get(size)
            if entry is None:
                # Skip if this backend didn't report this size
                continue
            mean, std = entry
            xs.append(xj[i])
            heights.append(mean)
            errs.append(std)

        if xs:
            plt.bar(xs, heights, width=width, yerr=errs, capsize=4, label=backend)

    # X axis as categorical sizes (more readable for grouped bars)
    plt.xticks(x, [str(s) for s in used_fft_sizes])
    plt.xlabel('Convolution Size (FFT size)')
    plt.ylabel('GB/s (higher is better)')
    plt.title('Convolution Performance Comparison (Grouped Bars)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    print(f'Saved {outfile}')

if __name__ == '__main__':
    merged, backends, fft_sizes = read_bench_csvs()

    print('\nSummary:')
    print(f'Backends found: {sorted(backends)}')
    print(f'Convolution sizes found: {sorted(fft_sizes)}')
    print(f'Total entries: {sum(len(v) for v in merged.values())}')

    sorted_backends = sorted(backends)
    sorted_fft_sizes = sorted(fft_sizes)

    # Grouped bar chart (side-by-side per size)
    save_grouped_bar_graph(["torch", "cufft", "zipfft", "vkdispatch"], sorted_fft_sizes, merged)
