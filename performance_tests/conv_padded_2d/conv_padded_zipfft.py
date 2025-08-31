import csv
import time
import performance_tests.conv_2d.conv_utils as fu
import numpy as np
import torch

try:
    from zipfft import cfft1d
    from zipfft import cfft1d_strided
except ImportError:
    print("zipfft is not installed. Please install it via 'pip install zipfft'.")
    exit(0)

def run_zipfft(config: fu.Config, fft_size: int) -> float:
    shape = config.make_shape(fft_size)
    random_data = config.make_random_data(fft_size)

    buffer = torch.empty(
        shape,
        dtype=torch.complex64,
        device='cuda'
    )

    buffer.copy_(torch.from_numpy(random_data).to('cuda'))

    stream = torch.cuda.Stream()

    torch.cuda.synchronize()
    
    with torch.cuda.stream(stream):
        for _ in range(config.warmup):
            cfft1d.fft(buffer)

    torch.cuda.synchronize()

    gb_byte_count = 11 * np.prod(shape) * 8 / (1024 * 1024 * 1024)
    
    g = torch.cuda.CUDAGraph()

    # We capture either 1 or K FFTs back-to-back. All on the same stream.
    with torch.cuda.graph(g, stream=stream):
        for _ in range(max(1, config.iter_batch)):
            cfft1d.fft(buffer)

    torch.cuda.synchronize()

    gb_byte_count = 2 * np.prod(shape) * 8 / (1024 * 1024 * 1024)
    
    start_time = time.perf_counter()

    for _ in range(config.iter_count // max(1, config.iter_batch)):
        g.replay()

    torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    return config.iter_count * gb_byte_count / elapsed_time

if __name__ == "__main__":
    config = fu.parse_args()
    fft_sizes = fu.get_fft_sizes()

    if config.axis != 1:
        print("zipfft currently only supports axis=1. Please set axis to 1.")
        exit(0)

    output_name = f"conv_padded_zipfft_{config.axis}_axis.csv"
    with open(output_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Backend', 'FFT Size'] + [f'Run {i + 1} (GB/s)' for i in range(config.run_count)] + ['Mean', 'Std Dev'])
        
        for fft_size in fft_sizes:
            rates = []

            for _ in range(config.run_count):
                gb_per_second = run_zipfft(config, fft_size)
                print(f"FFT Size: {fft_size}, Throughput: {gb_per_second:.2f} GB/s")
                rates.append(gb_per_second)

            rounded_data = [round(rate, 2) for rate in rates]
            rounded_mean = round(np.mean(rates), 2)
            rounded_std = round(np.std(rates), 2)

            writer.writerow(["zipfft", fft_size] + rounded_data + [rounded_mean, rounded_std])
        
    print(f"Results saved to {output_name}.csv")