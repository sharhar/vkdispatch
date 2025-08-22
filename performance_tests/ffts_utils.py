import sys
from typing import Tuple
import dataclasses

@dataclasses.dataclass
class Config:
    data_size: int
    axis: int
    iter_count: int
    iter_batch: int
    run_count: int
    warmup: int = 10

    def make_shape(self, fft_size: int) -> Tuple[int, ...]:
        shape = [0, 0]

        batched_axis = (self.axis + 1) % 2

        shape[self.axis] = fft_size
        shape[batched_axis] = self.data_size // fft_size

        return tuple(shape)

def parse_args() -> Config:
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <data_size> <axis> <iter_count> <iter_batch> <run_count>")
        sys.exit(1)

    return Config(
        data_size=int(sys.argv[1]),
        axis=int(sys.argv[2]),
        iter_count=int(sys.argv[3]),
        iter_batch=int(sys.argv[4]),
        run_count=int(sys.argv[5]),
    )

def get_fft_sizes():
    return [2**i for i in range(6, 13)]  # FFT sizes from 64 to 4096 (inclusive)

reference_list = []

def register_object(obj):
    reference_list.append(obj)
