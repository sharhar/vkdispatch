# Full end-to-end example:
# - PyTorch tensor storage is shared with vkdispatch via __cuda_array_interface__
# - vkdispatch kernel execution is captured inside torch.cuda.CUDAGraph
# - push-constant value ("scale") is updated between graph replays
# - a Const[...] argument ("bias") demonstrates UBO packing during capture (static in this example)

import torch

import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import Buff, Const, Var, f32


def main():
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    # Initialize vkdispatch with the PyCUDA backend and create a context on the same CUDA device.
    vd.initialize(backend="pycuda")
    vd.make_context(device_ids=torch.cuda.current_device())

    # Define a simple kernel:
    # y[i] = x[i] * scale + bias
    #
    # - scale: Var[f32]  -> push constant (mutable post-record via graph.set_var)
    # - bias:  Const[f32] -> uniform/constant (packed into UBO path)
    @vd.shader(exec_size=lambda args: args.x.size)
    def affine(y: Buff[f32], x: Buff[f32], scale: Var[f32], bias: Const[f32]):
        tid = vc.global_invocation_id().x
        y[tid] = x[tid] * scale + bias

    # Static tensors are important for CUDA Graph replay (pointer addresses must remain stable).
    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    # Zero-copy alias the tensors as vkdispatch buffers via __cuda_array_interface__.
    bx = vd.from_cuda_array(x)
    by = vd.from_cuda_array(y)

    # Build and record a vkdispatch command graph.
    # Use graph.bind_var("scale") to bind the push-constant slot to a named variable.
    cmd_graph = vd.CommandGraph()
    bias_value = 0.25  # This is Const[f32] (UBO-backed in this path), kept static in this example.

    affine(
        y=by,
        x=bx,
        scale=cmd_graph.bind_var("scale"),
        bias=bias_value,
        graph=cmd_graph,
    )

    # Set initial push-constant value before capture.
    cmd_graph.set_var("scale", 2.0)

    # Prepare capture resources (persistent staging, PC scratch, etc.) and pack current args.
    cap = cmd_graph.prepare_cuda_capture(instance_count=1)
    cmd_graph.update_captured_args(cap)

    # Capture vkdispatch submission into a torch CUDA graph.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # Submit on the same CUDA stream torch is capturing.
        cmd_graph.submit(cuda_stream=torch.cuda.current_stream(), capture=cap)

    # The capture run has executed once; validate it.
    torch.cuda.synchronize()
    expected = x * 2.0 + bias_value
    assert torch.allclose(y, expected, atol=1e-5, rtol=1e-5), "Initial captured run mismatch"

    # Replay with different push-constant values.
    for scale_value in [3.0, -1.5, 0.5]:
        cmd_graph.set_var("scale", scale_value)
        cmd_graph.update_captured_args(cap)  # updates persistent PC/UBO staging used by the captured graph
        g.replay()

        torch.cuda.synchronize()
        expected = x * scale_value + bias_value
        assert torch.allclose(y, expected, atol=1e-5, rtol=1e-5), f"Replay mismatch for scale={scale_value}"

    print("CUDA graph capture + replay with vkdispatch succeeded.")


if __name__ == "__main__":
    main()
