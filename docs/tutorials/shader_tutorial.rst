Shader Authoring and Dispatch
=============================

vkdispatch lets you write compute logic in Python syntax and compile it to GLSL at
runtime. This page covers the common shader workflow and launch patterns.

Examples below omit ``vd.initialize()`` and ``vd.make_context()`` because vkdispatch
creates them automatically on first runtime use. Call them manually only when you need
custom initialization/context settings.

Imports and Type Annotations
----------------------------

Most shader examples use these imports:

.. code-block:: python

   import vkdispatch as vd
   import vkdispatch.codegen as vc
   from vkdispatch.codegen.abreviations import *

* ``Buff[...]`` is a shader buffer argument type.
* ``Const[...]`` is a uniform/constant argument type.
* Dtype aliases such as ``f32``, ``i32``, and ``v2`` come from abbreviations.

Basic In-Place Kernel
---------------------

.. code-block:: python

   import numpy as np
   import vkdispatch as vd
   import vkdispatch.codegen as vc
   from vkdispatch.codegen.abreviations import *

   @vd.shader("buff.size")
   def add_scalar(buff: Buff[f32], bias: Const[f32]):
       tid = vc.global_invocation_id().x
       buff[tid] = buff[tid] + bias

   arr = np.arange(32, dtype=np.float32)
   buff = vd.asbuffer(arr)
   add_scalar(buff, 1.5)

   result = buff.read(0)
   print(result[:4])  # [1.5 2.5 3.5 4.5]

Launch Configuration
--------------------

Use one of these launch patterns:

* String expression (evaluated from function argument names):

  .. code-block:: python

     @vd.shader("in_buf.size")
     def kernel(in_buf: Buff[f32], out_buf: Buff[f32]):
         ...

* Fixed total dispatch size:

  .. code-block:: python

     @vd.shader(exec_size=(1024, 1, 1))
     def kernel(...):
         ...

* Dynamic size from call arguments:

  .. code-block:: python

     @vd.shader(exec_size=lambda args: args.in_buf.size)
     def kernel(in_buf: Buff[f32], out_buf: Buff[f32]):
         ...

* Explicit workgroups instead of ``exec_size``:

  .. code-block:: python

     @vd.shader(workgroups=(64, 1, 1), local_size=(128, 1, 1))
     def kernel(...):
         ...

``exec_size`` and ``workgroups`` are mutually exclusive.
The string form is often the most concise option for argument-dependent dispatch size.

Mapping Functions
-----------------

Mapping functions are reusable typed snippets (often used with reductions and FFT I/O).

.. code-block:: python

   @vd.map
   def square_value(x: Buff[f32]) -> f32:
       idx = vd.reduce.mapped_io_index()
       return x[idx] * x[idx]

You can pass mapping functions into APIs that accept ``mapping_function``,
``input_map``, or ``output_map`` arguments.

Inspecting Generated Shader Source
----------------------------------

A built shader can be printed for debugging:

.. code-block:: python

   print(add_scalar)

This prints GLSL-like generated source with line numbers, which is useful when debugging
type issues or unsupported expressions.

Common Notes
------------

* All shader parameters must be type annotated.
* Buffer/image arguments must use codegen types (for example, ``Buff[f32]``, ``Img2[f32]``).
* If you need batched submissions, prefer :doc:`Command Graph Recording <command_graph_tutorial>`.

Shader API Reference
--------------------

See the :doc:`Full Python API Reference <../python_api>` for complete API details on:

* ``vkdispatch.shader``
* ``vkdispatch.map``
* ``vkdispatch.ShaderFunction``
* ``vkdispatch.MappingFunction``
