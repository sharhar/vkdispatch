import vkdispatch as vd
import numpy as np

def test_fft_1d():
    max_fft_size = vd.get_context().max_shared_memory // vd.complex64.item_size

    max_fft_size = min(max_fft_size, vd.get_context().max_workgroup_size[0] * 8)

    current_fft_size = 2

    while current_fft_size <= max_fft_size:
        for _ in range(20):
            batch_size = np.random.randint(1, 2000)

            data = np.random.rand(batch_size, current_fft_size).astype(np.complex64)
            test_data = vd.asbuffer(data)

            vd.fft.fft(test_data)

            assert np.allclose(np.fft.fft(data, axis=1), test_data.read(0), atol=1e-3)

        current_fft_size *= 2
