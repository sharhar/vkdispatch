import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

def test_basic():
    graph = vd.CommandGraph()

    @vd.shader(exec_size=lambda args: args.buff.size)
    def test_shader(buff: Buff[f32], A: Const[f32]):
        tid = vc.global_invocation().x

        buff[tid] = buff[tid] + A

    signal = np.arange(32, dtype=np.float32)

    buff = vd.Buffer((32,) , vd.float32)
    buff.write(signal)

    test_shader(buff, 1.0, graph=graph)
    test_shader(buff, 2.0, graph=graph)

    graph.submit()

    assert np.allclose(buff.read(0), signal + 3, atol=0.00025)