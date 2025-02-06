import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

def test_arithmetic():
    pass_count = 64

    for _ in range(pass_count):
        array_size = np.random.randint(1000, 10000)

        signal = np.random.rand(array_size).astype(np.float32)
        signal2 = np.random.rand(array_size).astype(np.float32)

        buffer = vd.asbuffer(signal)
        buffer2 = vd.asbuffer(signal2)

        repeat_count = np.random.randint(10, 64)
        
        for _ in range(repeat_count):
            op_count = np.random.randint(2, 32)
            
            @vd.shader(exec_size=lambda args: args.a.size)
            def my_shader(a: Buff[f32], b: Buff[f32]):
                nonlocal signal, signal2

                tid = vc.global_invocation().x

                out_val = a[tid].copy()
                other_val = b[tid].copy()
                
                for _ in range(op_count):
                    op_number = np.random.randint(0, 4)

                    if op_number == 0:
                        out_val[:] = out_val + other_val
                        signal = signal + signal2
                    elif op_number == 1:
                        out_val[:] = out_val - other_val
                        signal = signal - signal2
                    elif op_number == 2:
                        out_val[:] = out_val * other_val
                        signal = signal * signal2
                    elif op_number == 3:
                        out_val[:] = out_val * vc.sin(other_val)
                        signal = signal * np.sin(signal2).astype(np.float32)
                
                a[tid] = out_val

            my_shader(buffer, buffer2)

        assert np.allclose(buffer.read(0), signal, atol=0.000025)
