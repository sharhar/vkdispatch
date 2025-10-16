import sys
from typing import Tuple
import dataclasses

import numpy as np

@dataclasses.dataclass
class Config:
    data_size: int
    iter_count: int
    iter_batch: int
    run_count: int
    warmup: int = 10

    def make_shape(self, fft_size: int) -> Tuple[int, ...]:
        total_square_size = fft_size * fft_size
        assert self.data_size % total_square_size == 0, "Data size must be a multiple of fft_size squared"
        return (self.data_size // total_square_size, fft_size, fft_size)
    
    def make_shape_2d(self, fft_size: int) -> Tuple[int, ...]:
        assert self.data_size % fft_size == 0, "Data size must be a multiple of fft_size squared"
        return (self.data_size // fft_size, fft_size)
    
    def make_random_data(self, fft_size: int):
        shape = self.make_shape(fft_size)
        return np.random.rand(*shape).astype(np.complex64)
    
    def make_random_data_2d(self, fft_size: int):
        shape = self.make_shape_2d(fft_size)
        return np.random.rand(*shape).astype(np.complex64)

def parse_args() -> Config:
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <data_size> <iter_count> <iter_batch> <run_count>")
        sys.exit(1)

    return Config(
        data_size=int(sys.argv[1]),
        iter_count=int(sys.argv[2]),
        iter_batch=int(sys.argv[3]),
        run_count=int(sys.argv[4]),
    )

def get_fft_sizes():
    return [2**i for i in range(6, 13)]  # FFT sizes from 64 to 4096 (inclusive)

