import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

def test_basic():
    vd.set_log_level(vd.LogLevel.VERBOSE)

    cmd_stream = vd.CommandStream()

    @vd.shader(exec_size=lambda args: args.buff.size)
    def test_shader(buff: Buff[f32], A: Const[f32]):
        tid = vc.global_invocation().x

        buff[tid] = buff[tid] + A

    signal = np.arange(32, dtype=np.float32)

    buff = vd.Buffer((32,) , vd.float32)
    buff.write(signal)

    test_shader(buff, 1.0, cmd_stream=cmd_stream)
    test_shader(buff, 2.0, cmd_stream=cmd_stream)

    cmd_stream.submit()

    assert np.allclose(buff.read(0), signal + 3, atol=0.00025)