import csv
import time
import ffts_utils as fu
import numpy as np
import torch

def run_torch(config: fu.Config, fft_size: int) -> float:
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
            buffer = torch.fft.fft2(buffer)

    torch.cuda.synchronize()

    gb_byte_count = 4 * np.prod(shape) * 8 / (1024 * 1024 * 1024)
    
    g = torch.cuda.CUDAGraph()

    # We capture either 1 or K FFTs back-to-back. All on the same stream.
    with torch.cuda.graph(g, stream=stream):
        for _ in range(max(1, config.iter_batch)):
            buffer = torch.fft.fft2(buffer)   # creates a tensor once during capture

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.cuda.stream(stream):
        for _ in range(config.iter_count // max(1, config.iter_batch)):
            g.replay()

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    return config.iter_count * gb_byte_count / elapsed_time

if __name__ == "__main__":
    config = fu.parse_args()
    fft_sizes = fu.get_fft_sizes()

    output_name = f"fft_torch.csv"
    with open(output_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Backend', 'FFT Size'] + [f'Run {i + 1} (GB/s)' for i in range(config.run_count)] + ['Mean', 'Std Dev'])
        
        for fft_size in fft_sizes:
            rates = []

            for _ in range(config.run_count):
                gb_per_second = run_torch(config, fft_size)
                print(f"FFT Size: {fft_size}, Throughput: {gb_per_second:.2f} GB/s")
                rates.append(gb_per_second)

            rounded_data = [round(rate, 2) for rate in rates]
            rounded_mean = round(np.mean(rates), 2)
            rounded_std = round(np.std(rates), 2)

            writer.writerow(["torch", fft_size] + rounded_data + [rounded_mean, rounded_std])
        
    print(f"Results saved to {output_name}.csv")