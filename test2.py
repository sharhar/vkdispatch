import vkdispatch as vd
import vkdispatch.codegen as vc

#vd.initialize(debug_mode=True, backend="cuda")
#vc.set_codegen_backend("cuda")

from typing import Callable, Union, Tuple

import numpy as np

import time
import dataclasses

@dataclasses.dataclass
class Config:
    data_size: int
    iter_count: int
    iter_batch: int
    run_count: int
    signal_factor: int
    warmup: int = 10

    def make_shape(self, fft_size: int) -> Tuple[int, ...]:
        total_square_size = fft_size * fft_size
        assert self.data_size % total_square_size == 0, "Data size must be a multiple of fft_size squared"
        return (self.data_size // total_square_size, fft_size, fft_size)
    
    def make_random_data(self, fft_size: int):
        shape = self.make_shape(fft_size)
        return np.random.rand(*shape).astype(np.complex64)

def run_vkdispatch(config: Config,
                    fft_size: int,
                    io_count: Union[int, Callable],
                    gpu_function: Callable) -> float:
    shape = config.make_shape(fft_size)

    buffer = vd.Buffer(shape, var_type=vd.complex64)
    kernel = vd.Buffer(shape, var_type=vd.complex64)

    graph = vd.CommandGraph()
    old_graph = vd.set_global_graph(graph)
    
    gpu_function(config, fft_size, buffer, kernel)

    vd.set_global_graph(old_graph)

    for _ in range(config.warmup):
        graph.submit(config.iter_batch)

    vd.queue_wait_idle()

    if callable(io_count):
        io_count = io_count(buffer.size, fft_size)

    gb_byte_count = io_count * 8 * buffer.size / (1024 * 1024 * 1024)
    
    start_time = time.perf_counter()

    for _ in range(config.iter_count // config.iter_batch):
        graph.submit(config.iter_batch)

    vd.queue_wait_idle()

    elapsed_time = time.perf_counter() - start_time

    buffer.destroy()
    kernel.destroy()
    graph.destroy()
    vd.fft.cache_clear()

    time.sleep(1)

    vd.queue_wait_idle()    

    return gb_byte_count, elapsed_time


def run_test(config: Config,
               io_count: Union[int, Callable],
               gpu_function: Callable):
    fft_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    for fft_size in fft_sizes:
        rates = []

        for _ in range(config.run_count):
            gb_byte_count, elapsed_time = run_vkdispatch(config, fft_size, io_count, gpu_function)
            gb_per_second = config.iter_count * gb_byte_count / elapsed_time

            print(f"FFT Size: {fft_size}, Throughput: {gb_per_second:.4f} GB/s")
            rates.append(gb_per_second)

def do_fft(config: Config,
                    fft_size: int,
                    buffer: vd.Buffer,
                    kernel: vd.Buffer):
    vd.fft.fft(buffer)


conf = Config(
    data_size=2**26,
    iter_count=80,
    iter_batch=10,
    run_count=1,
    signal_factor=8
)

run_test(conf, 2, do_fft)