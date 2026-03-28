Code Structure and Execution Flow
=================================

This page explains how the vkdispatch repository is organized and how a Python call
is translated into GPU work. If you are extending the project or debugging behavior,
this should be your first stop.

In normal usage, ``vkdispatch`` will call ``initialize()`` and ``make_context()``
automatically the first time you invoke most runtime APIs. You only need to call
them manually if you want non-default settings (for example debug logging, custom
device selection, or multi-queue behavior).

Repository Layout
-----------------

Top-level folders you will use most often:

* ``vkdispatch/``: Public Python API and high-level runtime logic.
* ``vkdispatch_native/``: Native C++/Cython backend called by the Python layer.
* ``tests/``: End-to-end usage examples and regression coverage.
* ``docs/``: Sphinx docs (this site).
* ``deps/``: Third-party dependencies used for source builds.

Python Package Layout
---------------------

Inside ``vkdispatch/``, modules are grouped by responsibility:

* ``vkdispatch/base``: Core runtime objects and Vulkan-facing wrappers.
  
  * ``init.py``: Vulkan instance/device discovery and initialization.
  * ``context.py``: Global context creation, queue/device selection, lifecycle.
  * ``buffer.py`` / ``image.py``: GPU data containers.
  * ``compute_plan.py`` / ``descriptor_set.py`` / ``command_list.py``: Low-level execution objects.

* ``vkdispatch/shader``: Python-to-shader front-end.
  
  * ``decorator.py``: ``@vd.shader`` entry point.
  * ``signature.py``: Type-annotated argument parsing and shader signature building.
  * ``shader_function.py``: Build, bind, and dispatch compiled shader functions.
  * ``map.py``: Mapping-function abstraction shared by FFT/reduction paths.

* ``vkdispatch/codegen``: GLSL code generation utilities and typed shader variables.

* ``vkdispatch/execution_pipeline``: Higher-level command recording.
  
  * ``command_graph.py``: ``CommandGraph`` wrapper over ``CommandList`` with automatic buffer/constant management.

* ``vkdispatch/reduce``: Reduction decorators and staged reduction pipeline generation.

* ``vkdispatch/fft`` and ``vkdispatch/vkfft``: FFT/convolution front-ends.
  
  * ``fft``: vkdispatch shader-generated FFT path.
  * ``vkfft``: VkFFT-backed path with plan caching.

Native Backend Layout
---------------------

The compiled extension module is built from ``vkdispatch_native/``:

* ``wrapper.pyx``: Cython bridge exposing native entry points to Python.
* ``context/``: Device/context creation and global state.
* ``objects/``: Native Buffer/Image/DescriptorSet/CommandList objects.
* ``stages/``: Compute/FFT stage planning and recording.
* ``queue/``: Queue management, signals, and barriers.
* ``libs/``: Third-party integration glue (Volk, VMA).

During execution, most Python API methods forward to ``vkdispatch_native`` and then
call error checks to surface native failures as Python exceptions.

End-to-End Runtime Flow
-----------------------

Typical call path for a shader dispatch:

1. First vkdispatch runtime call triggers ``initialize()`` and ``make_context()`` (unless you called them manually first).
2. ``@vd.shader`` wraps a Python function and records typed operations via ``vkdispatch.codegen``.
3. ``ShaderFunction.build()`` generates GLSL and creates a ``ComputePlan``.
4. A ``CommandGraph`` (default or explicit) records bindings and dispatch dimensions.
5. ``CommandGraph.submit()`` submits the command list to selected queue(s).
6. Data is read back with ``Buffer.read()`` or ``Image.read()``.

Minimal Example (API Layer View)
--------------------------------

.. code-block:: python

   import numpy as np
   import vkdispatch as vd
   import vkdispatch.codegen as vc
   from vkdispatch.codegen.abreviations import *

   # @vd.shader(exec_size=lambda args: args.data.size)
   @vd.shader("data.size")
   def scale_inplace(data: Buff[f32], alpha: Const[f32]):
       tid = vc.global_invocation_id().x
       data[tid] = data[tid] * alpha

   arr = np.arange(16, dtype=np.float32)
   buf = vd.asbuffer(arr)
   scale_inplace(buf, 2.0)

   out = buf.read(0)
   print(out)  # [0, 2, 4, ...]

Related Tutorials
-----------------

* :doc:`Context System <context_system>`
* :doc:`Shader Authoring and Dispatch <shader_tutorial>`
* :doc:`Command Graph Recording <command_graph_tutorial>`
