# vkdispatch

`vkdispatch` is a Python GPU computing framework for writing single-source kernels in Python and dispatching them across multiple runtime backends.

It combines runtime code generation, execution helpers, and FFT/reduction utilities in one package. The default PyPI install ships with the Vulkan backend. CUDA and OpenCL backends can be enabled with optional runtime dependencies.

## Highlights

- Single-source Python shaders via `@vd.shader` and `vkdispatch.codegen`
- Multiple runtime backends: Vulkan, CUDA, OpenCL, and a dummy codegen-only backend
- Backend-aware code generation: GLSL for Vulkan, CUDA source for CUDA, and OpenCL C for OpenCL
- Native FFT workflows through `vd.fft`, including mapping hooks for fusion and custom I/O
- VkFFT-backed transforms through `vd.vkfft` on the Vulkan backend
- Reductions through `vd.reduce`
- Batched submission and deferred execution through `vd.CommandGraph`
- CUDA interop through `__cuda_array_interface__` and CUDA Graph capture helpers

## Installation

### Default Vulkan Install

To install `vkdispatch` with the Vulkan backend, run:

```bash
pip install vkdispatch
```

This installs the core library, the code generation system, and the Vulkan runtime backend. The Vulkan backend is designed to run on systems supporting Vulkan 1.2 or higher, including macOS via a statically linked MoltenVK. Alternate backends can be added with optional dependencies as described below.

On mainstream platforms - Windows (`x86_64`), macOS (`x86_64` and Apple Silicon/`arm64`), and Linux (`x86_64`) - `pip` will usually download a prebuilt wheel, so no compiler is needed.

On less common platforms, `pip` may fall back to a source build, which takes a few minutes. See [Building From Source](https://sharhar.github.io/vkdispatch/tutorials/building_from_source.html) for toolchain requirements and developer-oriented instructions.

### Core package

For cases where only the codegen component is needed, or in environments where only the CUDA or OpenCL backends are needed, install the core package:

```bash
pip install vkdispatch-core
```

This installs the core library and codegen components, but not the Vulkan runtime backend. To enable runtime features beyond pure codegen, install the optional dependencies below.

### Optional components

- Optional CLI: `pip install "vkdispatch-core[cli]"`
- CUDA runtime backend: `pip install "vkdispatch-core[cuda]`
- OpenCL runtime backend: `pip install "vkdispatch-core[opencl]`

## Runtime backends

`vkdispatch` currently supports these runtime backends:

- `vulkan`
- `cuda`
- `opencl`
- `dummy`

If you do not explicitly select a backend, ``vkdispatch`` prefers Vulkan. When the Vulkan backend cannot be imported because it is not installed, initialization falls back to CUDA and then OpenCL.

You can select a backend explicitly in Python:

```python
import vkdispatch as vd

vd.initialize(backend="vulkan")
# vd.initialize(backend="cuda")
# vd.initialize(backend="opencl")
# vd.initialize(backend="dummy")
```

You can also select the backend with an environment variable:

```bash
export VKDISPATCH_BACKEND=vulkan
```

The dummy backend is useful for codegen-only workflows, source inspection, and development environments where no GPU runtime is available.

There are two intended shader-generation modes:

- Default mode: generate for the current machine/runtime. This is the normal path and is how `vkdispatch` picks backend-specific defaults and limits.
- Custom mode: initialize with `backend="dummy"` and optionally tune the dummy device limits when you want controlled codegen without relying on the current runtime.


## Verifying your installation

If you installed the optional CLI, you can list devices with:

```bash
vdlist

# Explicit backend selection can be done with cmdline flags:
vdlist --vulkan
vdlist --cuda
vdlist --opencl
```

You can always inspect devices from Python:

```python
import vkdispatch as vd

for device in vd.get_devices():
    print(device.get_info_string())
```

The reported version label depends on the active backend:

- Vulkan devices show a Vulkan version
- CUDA devices show CUDA compute capability
- OpenCL devices show an OpenCL version

## Quick start

The example below defines a simple in-place compute kernel in Python:

```python
import numpy as np
import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import Buff, Const, f32

# @vd.shader(exec_size=lambda args: args.buff.size)
@vd.shader("buff.size")
def add_scalar(buff: Buff[f32], bias: Const[f32]):
    tid = vc.global_invocation_id().x
    buff[tid] = buff[tid] + bias

arr = np.arange(8, dtype=np.float32)
buff = vd.asbuffer(arr)

# If you want a non-default backend, call vd.initialize(backend=...) first.
add_scalar(buff, 1.5)

print(buff.read(0))
```

String launch sizing is the shortest form and is kept for convenience. If you want
the launch rule to be more explicit and deterministic, use the equivalent lambda form
instead: `@vd.shader(exec_size=lambda args: args.buff.size)`.

In normal usage, `vkdispatch` initializes itself and creates a default context on first runtime use. Call `vd.initialize()` and `vd.make_context()` manually only when you want non-default settings such as backend selection, custom device selection, debug logging, or multi-device Vulkan contexts.

## Codegen-Only Workflows

If you want generated source without compiling or dispatching it on the current machine, use the dummy backend explicitly:

```python
import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import Buff, Const, f32

vd.initialize(backend='dummy')
vd.set_dummy_context_params(
    subgroup_size=32,
    max_workgroup_size=(128, 1, 1),
    max_workgroup_count=(65535, 65535, 65535),
)
vc.set_codegen_backend('cuda')

# @vd.shader(exec_size=lambda args: args.buff.size)
@vd.shader('buff.size')
def add_scalar(buff: Buff[f32], bias: Const[f32]):
    tid = vc.global_invocation_id().x
    buff[tid] = buff[tid] + bias

src = add_scalar.get_src(line_numbers=True)
print(src)
```

In this mode, `vkdispatch` uses the dummy device model for launch/layout defaults and emits source for the backend selected with `vc.set_codegen_backend(...)`.

## Documentation

The docs site is still under active development, but the main entry points are here:

- [Getting Started](https://sharhar.github.io/vkdispatch/getting_started.html)
- [Tutorials](https://sharhar.github.io/vkdispatch/tutorials/index.html)
- [Python API Reference](https://sharhar.github.io/vkdispatch/python_api.html)

Some especially useful tutorials:

- [Shader Authoring and Dispatch](https://sharhar.github.io/vkdispatch/tutorials/shader_tutorial.html)
- [Initialization and Context Creation](https://sharhar.github.io/vkdispatch/tutorials/context_system.html)
- [Command Graph Recording](https://sharhar.github.io/vkdispatch/tutorials/command_graph_tutorial.html)
- [Reductions and FFT Workflows](https://sharhar.github.io/vkdispatch/tutorials/reductions_and_fft.html)

Happy GPU programming!
