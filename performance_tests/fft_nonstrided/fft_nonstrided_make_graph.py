import glob
import csv
from typing import Dict, Tuple, Set
from matplotlib import pyplot as plt
import numpy as np
import sys

# Nested structure:
# merged[backend][fft_size] = (mean, std)
MergedType = Dict[str, Dict[int, Tuple[float, float]]]

def read_bench_csvs() -> Tuple[MergedType, Set[str], Set[int]]:
    pattern = f"fft_nonstrided_*.csv"
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

def save_graph(backends: Set[str], fft_sizes: Set[int], merged: MergedType, min_fft_size: int = None):
    plt.figure(figsize=(10, 6))

    if min_fft_size is not None:
        used_fft_sizes = [size for size in fft_sizes if size >= min_fft_size]
    else:
        used_fft_sizes = fft_sizes

    for backend_name in backends:
        means = [
            merged[backend_name][i][0]
            for i in used_fft_sizes
        ]
        stds = [
            merged[backend_name][i][1]
            for i in used_fft_sizes
        ]
        
        plt.errorbar(
            used_fft_sizes,
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
    if min_fft_size is not None:
        plt.savefig(f"fft_graph_min_size{min_fft_size}.png")
        return
    plt.savefig(f"fft_graph.png")

if __name__ == "__main__":
    # Example usage (change the number as needed)
    merged, backends, fft_sizes = read_bench_csvs()

    print("\nSummary:")
    print(f"Backends found: {sorted(backends)}")
    print(f"FFT sizes found: {sorted(fft_sizes)}")
    print(f"Total entries: {sum(len(v) for v in merged.values())}")

    sorted_backends = sorted(backends)
    sorted_fft_sizes = sorted(fft_sizes)

    save_graph(sorted_backends, sorted_fft_sizes, merged)
    save_graph(sorted_backends, sorted_fft_sizes, merged, min_fft_size=256)

    
