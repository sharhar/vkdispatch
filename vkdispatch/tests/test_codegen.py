import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

def make_random_complex_signal(shape):
    r = np.random.random(size=shape)
    i = np.random.random(size=shape)
    return (r + i * 1j).astype(np.complex64)

def test_arithmetic():
    # Create a 1D buffer
    signal = make_random_complex_signal((50,))
    signal2 = make_random_complex_signal((50,))
    signal3 = make_random_complex_signal((50,))
    
    test_line = vd.Buffer((signal.shape[0],), vd.complex64)
    test_line2 = vd.Buffer((signal2.shape[0],), vd.complex64)
    test_line3 = vd.Buffer((signal3.shape[0],), vd.complex64)
    
    test_line.write(signal)
    test_line2.write(signal2)

    @vc.shader(exec_size=lambda args: args.out.size)
    def add_buffers(out: Buff[c64], a: Buff[c64], b: Buff[c64]):
        tid = vc.global_invocation.x
        out[tid] = a[tid] + b[tid]

    add_buffers(test_line3, test_line, test_line2)

    assert np.allclose(test_line3.read(0), signal + signal2, atol=0.00001)