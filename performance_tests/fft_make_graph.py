import glob
import csv
from typing import Dict, Tuple, Set
from matplotlib import pyplot as plt
import numpy as np
import sys

# Nested structure:
# merged[backend][fft_size] = (mean, std)
MergedType = Dict[str, Dict[int, Tuple[float, float]]]

def read_bench_csvs(number: int) -> Tuple[MergedType, Set[str], Set[int]]:
    pattern = f"fft_*_{number}_axis.csv"
    files = glob.glob(pattern)

    merged: MergedType = {}
    backends: Set[str] = set()
    fft_sizes: Set[int] = set()

    for filename in files:
        print(f"Reading: {filename}")
        with open(filename, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                backend = row["Backend"].strip()
                size = int(row["FFT Size"])
                mean = float(row["Mean"])
                std = float(row["Std Dev"])

                backends.add(backend)
                fft_sizes.add(size)

                if backend not in merged:
                    merged[backend] = {}

                # last one wins if duplicates appear across files
                merged[backend][size] = (mean, std)

    return merged, backends, fft_sizes

if __name__ == "__main__":
    axis = int(sys.argv[1])

    # Example usage (change the number as needed)
    merged, backends, fft_sizes = read_bench_csvs(axis)

    print("\nSummary:")
    print(f"Backends found: {sorted(backends)}")
    print(f"FFT sizes found: {sorted(fft_sizes)}")
    print(f"Total entries: {sum(len(v) for v in merged.values())}")

    sorted_backends = sorted(backends)
    sorted_fft_sizes = sorted(fft_sizes)

    plt.figure(figsize=(10, 6))
    for backend_name in sorted_backends:
        means = [
            merged[backend_name][i][0]
            for i in sorted_fft_sizes
        ]
        stds = [
            merged[backend_name][i][1]
            for i in sorted_fft_sizes
        ]
        
        plt.errorbar(
            sorted_fft_sizes,
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
    plt.savefig(f"fft_graph_axis_{axis}.png")
