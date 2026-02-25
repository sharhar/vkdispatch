#!/usr/bin/env python3
"""Capture and replay a vkdispatch CUDA kernel inside a PyTorch CUDA Graph.

This example uses:
  - vkdispatch runtime backend: "cuda"
  - a custom vkdispatch shader recorded into CommandGraph
  - torch.cuda.CUDAGraph capture + replay
  - zero-copy tensor sharing via __cuda_array_interface__
"""

import torch

import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import Buff, Const, f32


@vd.shader(exec_size=lambda args: args.x.size)
def custom_shader(out: Buff[f32], x: Buff[f32], bias: Const[f32]):
    tid = vc.global_invocation_id().x
    out[tid] = x[tid] * 1.5 + vc.sin(x[tid]) + bias


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example.")

    torch.cuda.set_device(0)
    torch.manual_seed(0)

    vd.initialize(backend="cuda")
    vd.make_context(device_ids=torch.cuda.current_device())

    n = 16
    bias = 0.25

    # Static allocations are required for CUDA Graph replay.
    x = torch.empty(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    x.fill_(0.0)

    x_vd = vd.from_cuda_array(x)
    out_vd = vd.from_cuda_array(out)

    cmd_graph = vd.CommandGraph()

    # Record one vkdispatch kernel launch into the command graph.
    # For backend="cuda-python", Const/Var payloads are fixed at record time.
    custom_shader(out=out_vd, x=x_vd, bias=bias, graph=cmd_graph)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cmd_graph.submit(cuda_stream=torch.cuda.current_stream())

    replay_inputs = [0.0, 1.0, 2.0, 3.0]
    for i, value in enumerate(replay_inputs, start=1):
        x.fill_(value)
        graph.replay()
        torch.cuda.synchronize()

        expected = x * 1.5 + torch.sin(x) + bias
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        print(
            f"replay {i} input={value:.1f} output[:8]={out[:8].detach().cpu().tolist()}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
